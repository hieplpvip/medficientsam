#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor-io/xnpz.hpp>

#include "lrucache.hpp"

using namespace std::string_literals;
using ImageSize = std::array<size_t, 2>;

constexpr size_t EMBEDDINGS_CACHE_SIZE = 1024;
constexpr size_t IMAGE_ENCODER_INPUT_SIZE = 512;
const ov::Shape INPUT_SHAPE = {IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE, 3};

std::array<size_t, 2> get_preprocess_shape(size_t oldh, size_t oldw) {
  double scale = 1.0 * IMAGE_ENCODER_INPUT_SIZE / std::max(oldh, oldw);
  size_t newh = scale * oldh + 0.5;
  size_t neww = scale * oldw + 0.5;
  return {newh, neww};
}

xt::xtensor<float, 1> get_bbox(xt::xtensor<float, 2>& mask) {
  auto indices = xt::where(mask > 0);
  auto y_indices = indices[0], x_indices = indices[1];
  auto x_min = *std::min_element(x_indices.begin(), x_indices.end());
  auto x_max = *std::max_element(x_indices.begin(), x_indices.end());
  auto y_min = *std::min_element(y_indices.begin(), y_indices.end());
  auto y_max = *std::max_element(y_indices.begin(), y_indices.end());
  return {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
}

template <class T>
T cast_npy_file(xt::detail::npy_file& npy_file) {
  auto m_typestring = npy_file.m_typestring;
  if (m_typestring == "|u1") {
    return npy_file.cast<uint8_t>();
  } else if (m_typestring == "<u2") {
    return npy_file.cast<uint16_t>();
  } else if (m_typestring == "<u4") {
    return npy_file.cast<uint32_t>();
  } else if (m_typestring == "<u8") {
    return npy_file.cast<uint64_t>();
  } else if (m_typestring == "|i1") {
    return npy_file.cast<int8_t>();
  } else if (m_typestring == "<i2") {
    return npy_file.cast<int16_t>();
  } else if (m_typestring == "<i4") {
    return npy_file.cast<int32_t>();
  } else if (m_typestring == "<i8") {
    return npy_file.cast<int64_t>();
  } else if (m_typestring == "<f4") {
    return npy_file.cast<float>();
  } else if (m_typestring == "<f8") {
    return npy_file.cast<double>();
  }
  XTENSOR_THROW(std::runtime_error, "Cast error: unknown format "s + m_typestring);
}

struct Encoder {
  ov::CompiledModel model;
  ov::InferRequest infer_request;
  ImageSize original_size, new_size;

  Encoder(ov::Core& core, const std::string& model_path) {
    model = core.compile_model(model_path, "CPU");
    infer_request = model.create_infer_request();
  }

  void set_sizes(const ImageSize& original_size, const ImageSize& new_size) {
    this->original_size = original_size;
    this->new_size = new_size;
  }

  ov::Tensor encode_image(const ov::Tensor& input_tensor) {
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    return infer_request.get_output_tensor();
  }

  xt::xtensor<float, 3> preprocess_2D(xt::xtensor<uint8_t, 3>& original_img) {
    assert(original_img.shape()[0] == 3);
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC3, original_img.data()), mat2;
    cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat2.data, mat2.total() * mat2.channels(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, mat2.channels()});
    img = (img - xt::amin(img)()) / std::clamp(xt::amax(img)() - xt::amin(img)(), 1e-8f, 1e18f);
    return xt::pad(img, {{0, IMAGE_ENCODER_INPUT_SIZE - new_size[0]}, {0, IMAGE_ENCODER_INPUT_SIZE - new_size[1]}, {0, 0}});
  }

  xt::xtensor<float, 3> preprocess_3D(xt::xtensor<uint8_t, 3>& original_img, int z) {
    auto data = original_img.data() + z * original_size[0] * original_size[1];
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC1, data), mat2;
    cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat2.data, mat2.total(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, 1});
    img = (img - xt::amin(img)()) / std::clamp(xt::amax(img)() - xt::amin(img)(), 1e-8f, 1e18f);
    return xt::repeat(xt::pad(img, {{0, IMAGE_ENCODER_INPUT_SIZE - new_size[0]}, {0, IMAGE_ENCODER_INPUT_SIZE - new_size[1]}, {0, 0}}), 3, 2);
  }
};

struct Decoder {
  ov::CompiledModel model;
  ov::InferRequest infer_request;
  ImageSize original_size, new_size;

  Decoder(ov::Core& core, const std::string& model_path) {
    model = core.compile_model(model_path, "CPU");
    infer_request = model.create_infer_request();
  }

  void set_sizes(const ImageSize& original_size, const ImageSize& new_size) {
    this->original_size = original_size;
    this->new_size = new_size;
  }

  void set_embedding_tensor(const ov::Tensor& embedding_tensor) {
    infer_request.set_input_tensor(0, embedding_tensor);
  }

  xt::xtensor<float, 2> decode_mask(const ov::Tensor& box_tensor) {
    infer_request.set_input_tensor(1, box_tensor);
    infer_request.infer();

    xt::xtensor<float, 2> mask = xt::adapt(infer_request.get_output_tensor().data<float>(), IMAGE_ENCODER_INPUT_SIZE * IMAGE_ENCODER_INPUT_SIZE, xt::no_ownership(), std::vector<int>{IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE});
    mask = xt::view(mask, xt::range(_, new_size[0]), xt::range(_, new_size[1]));

    cv::Mat mat1(cv::Size(new_size[1], new_size[0]), CV_32FC1, mask.data()), mat2;
    cv::resize(mat1, mat2, cv::Size(original_size[1], original_size[0]), cv::INTER_LINEAR);
    return xt::adapt((float*)mat2.data, mat2.total(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols});
  }
};

void infer_2d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<float, 2>>(npz_data["boxes"]);
  assert(original_img.shape()[0] == 3);
  assert(boxes.shape()[1] == 4);

  ImageSize original_size = {original_img.shape()[0], original_img.shape()[1]};
  ImageSize new_size = get_preprocess_shape(original_size[0], original_size[1]);
  boxes /= std::max(original_size[0], original_size[1]);
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);

  auto img = encoder.preprocess_2D(original_img);
  ov::Tensor input_tensor(ov::element::f32, INPUT_SHAPE, img.data());

  // auto encoder_start = std::chrono::high_resolution_clock::now();
  ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
  // auto encoder_finish = std::chrono::high_resolution_clock::now();
  // std::cout << "Encoded image in " << std::chrono::duration_cast<std::chrono::milliseconds>(encoder_finish - encoder_start).count() << "ms\n";

  xt::xtensor<uint16_t, 2> segs = xt::zeros<uint16_t>({original_size[0], original_size[1]});

  decoder.set_embedding_tensor(embedding_tensor);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    ov::Tensor box_tensor(ov::element::f32, {4}, boxes.data() + i * 4);
    // auto decoder_start = std::chrono::high_resolution_clock::now();
    auto mask = decoder.decode_mask(box_tensor);
    // auto decoder_finish = std::chrono::high_resolution_clock::now();
    // std::cout << "Decoded box " << (i + 1) << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(decoder_finish - decoder_start).count() << "ms\n";
    xt::filtration(segs, mask > 0) = i + 1;
  }

  xt::dump_npz(seg_file, "segs", segs, true);
}

void infer_3d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<uint16_t, 2>>(npz_data["boxes"]);
  assert(boxes.shape()[1] == 6);

  ImageSize original_size = {original_img.shape()[1], original_img.shape()[2]};
  ImageSize new_size = get_preprocess_shape(original_size[0], original_size[1]);
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);

  cache::lru_cache<int, ov::Tensor> cached_embeddings(EMBEDDINGS_CACHE_SIZE);
  auto get_embedding = [&](int z) {
    if (cached_embeddings.exists(z)) {
      return cached_embeddings.get(z);
    }
    auto img = encoder.preprocess_3D(original_img, z);
    ov::Tensor input_tensor(ov::element::f32, INPUT_SHAPE, img.data());
    ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
    cached_embeddings.put(z, embedding_tensor);
    return embedding_tensor;
  };
  auto process_slice = [&](int z, xt::xtensor<float, 1>& box) {
    ov::Tensor embedding_tensor = get_embedding(z);
    ov::Tensor box_tensor(ov::element::f32, {4}, box.data());
    decoder.set_embedding_tensor(embedding_tensor);
    return decoder.decode_mask(box_tensor);
  };

  xt::xtensor<uint16_t, 3> segs = xt::zeros_like(original_img);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    uint16_t x_min = boxes(i, 0), y_min = boxes(i, 1), z_min = boxes(i, 2);
    uint16_t x_max = boxes(i, 3), y_max = boxes(i, 4), z_max = boxes(i, 5);
    z_min = std::max(z_min, uint16_t(0));
    z_max = std::min(z_max, uint16_t(original_img.shape()[0]));
    uint16_t z_middle = (z_min + z_max) / 2;

    xt::xtensor<float, 1> box_default = {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
    box_default /= std::max(original_size[0], original_size[1]);

    // infer z_middle
    xt::xtensor<float, 1> box_middle;
    {
      auto mask_middle = process_slice(z_middle, box_default);
      xt::filtration(xt::view(segs, z_middle, xt::all(), xt::all()), mask_middle > 0) = i + 1;
      if (xt::amax(mask_middle)() > 0) {
        box_middle = get_bbox(mask_middle) / std::max(original_size[0], original_size[1]);
      } else {
        box_middle = box_default;
      }
    }

    // infer z_middle+1 to z_max-1
    auto last_box = box_middle;
    for (int z = z_middle + 1; z < z_max; ++z) {
      auto mask = process_slice(z, last_box);
      xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      if (xt::amax(mask)() > 0) {
        last_box = get_bbox(mask) / std::max(original_size[0], original_size[1]);
      } else {
        last_box = box_default;
      }
    }

    // infer z_min to z_middle-1
    last_box = box_middle;
    for (int z = z_middle - 1; z >= z_min; --z) {
      auto mask = process_slice(z, last_box);
      xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      if (xt::amax(mask)() > 0) {
        last_box = get_bbox(mask) / std::max(original_size[0], original_size[1]);
      } else {
        last_box = box_default;
      }
    }
  }

  xt::dump_npz(seg_file, "segs", segs, true);
}

bool starts_with(const std::string& str, const std::string& prefix) {
  return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0] << " <encoder.xml> <decoder.xml> <model cache folder> <imgs folder> <segs folder>\n";
    return 1;
  }

  ov::Core core;
  core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
  core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
  core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  core.set_property("CPU", ov::hint::num_requests(1));
  core.set_property(ov::cache_dir(argv[3]));
  Encoder encoder(core, argv[1]);
  Decoder decoder(core, argv[2]);

  std::filesystem::path imgs_folder(argv[4]);
  if (!std::filesystem::is_directory(imgs_folder)) {
    throw std::runtime_error(imgs_folder.string() + " is not a folder");
  }

  std::filesystem::path segs_folder(argv[5]);
  if (!std::filesystem::exists(segs_folder) && !std::filesystem::create_directory(segs_folder)) {
    throw std::runtime_error("Failed to create " + segs_folder.string());
  }
  if (!std::filesystem::is_directory(segs_folder)) {
    throw std::runtime_error(segs_folder.string() + " is not a folder");
  }

  for (const auto& entry : std::filesystem::directory_iterator(imgs_folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    auto base_name = entry.path().filename().string();
    if (ends_with(base_name, ".npz")) {
      auto img_file = entry.path().string();
      auto seg_file = (segs_folder / entry.path().filename()).string();

      std::cout << "Processing " << base_name << std::endl;
      auto infer_start = std::chrono::high_resolution_clock::now();
      if (starts_with(base_name, "2D")) {
        infer_2d(img_file, seg_file, encoder, decoder);
      } else {
        infer_3d(img_file, seg_file, encoder, decoder);
      }
      auto infer_finish = std::chrono::high_resolution_clock::now();
      std::cout << "Inferred " << base_name << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(infer_finish - infer_start).count() << "ms\n";
    }
  }

  return 0;
}
