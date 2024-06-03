FROM ubuntu:22.04 as build
USER root
WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN apt-get update && apt-get install -y ca-certificates git unzip wget

# Download OpenCV source code
ENV OPENCV_VERSION='4.9.0'
ENV OPENCV_SRC=/opencv-${OPENCV_VERSION}
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && unzip ${OPENCV_VERSION}.zip

# Download OpenVINO source code
ENV OPENVINO_REPO=/openvino
RUN git clone https://github.com/openvinotoolkit/openvino.git --branch 2024.0.0 --single-branch --recursive
RUN ${OPENVINO_REPO}/install_build_dependencies.sh

# Build OpenCV
COPY cpp/opencv4_cmake_options.txt opencv4_cmake_options.txt
RUN cmake \
    -DENABLE_LTO=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIBS="core,imgproc"\
    -DCPU_BASELINE=AVX2 \
    -DCPU_DISPATCH= \
    `cat opencv4_cmake_options.txt` \
    -S ${OPENCV_SRC} \
    -B ${OPENCV_SRC}/build
RUN cmake --build ${OPENCV_SRC}/build --target install --parallel 8

# Build OpenVINO
ENV OPENVINO_INSTALL_DIR=/opt/intel/openvino
# COPY cpp/fix_cache.patch ${OPENVINO_REPO}/fix_cache.patch
# RUN git -C ${OPENVINO_REPO} apply fix_cache.patch
# https://github.com/openvinotoolkit/openvino/blob/master/cmake/features.cmake
RUN cmake \
    -G "Ninja Multi-Config" \
    -DENABLE_LTO=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DENABLE_INTEL_CPU=ON \
    -DENABLE_INTEL_GPU=OFF \
    -DENABLE_MULTI=OFF \
    -DENABLE_AUTO=OFF \
    -DENABLE_AUTO_BATCH=OFF \
    -DENABLE_HETERO=OFF \
    -DENABLE_TEMPLATE=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_OV_ONNX_FRONTEND=OFF \
    -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_IR_FRONTEND=ON \
    -DENABLE_OV_PYTORCH_FRONTEND=OFF \
    -DENABLE_OV_TF_FRONTEND=OFF \
    -DENABLE_OV_TF_LITE_FRONTEND=OFF \
    -DENABLE_CPPLINT=OFF \
    -DENABLE_NCC_STYLE=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_STRICT_DEPENDENCIES=OFF \
    -DENABLE_SYSTEM_TBB=ON \
    -DENABLE_SYSTEM_OPENCL=ON \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCPACK_GENERATOR=TGZ \
    -S ${OPENVINO_REPO} \
    -B ${OPENVINO_BUILD_DIR}/build
RUN cmake --build ${OPENVINO_BUILD_DIR}/build --config Release --parallel 8
RUN cmake -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL_DIR} -P ${OPENVINO_BUILD_DIR}/build/cmake_install.cmake

RUN mkdir /medsam
WORKDIR /medsam

# Build app
COPY cpp/src src
COPY cpp/libs libs
COPY cpp/CMakeLists.txt CMakeLists.txt
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR=${OPENVINO_INSTALL_DIR}/runtime/cmake
RUN cmake --build build

FROM ubuntu:22.04 as main
USER root
WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libtbb12 time

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /medsam /inputs /outputs
RUN chown user:user /medsam /inputs /outputs

USER user
WORKDIR /medsam

COPY --chown=user:user --from=build /medsam/build/main main
COPY --chown=user:user cpp/predict.sh predict.sh

ARG EXPORTED_MODEL_PATH=weights/finetuned-l1-augmented/e2_cpp
COPY --chown=user:user ${EXPORTED_MODEL_PATH}/encoder.xml encoder.xml
COPY --chown=user:user ${EXPORTED_MODEL_PATH}/encoder.bin encoder.bin
COPY --chown=user:user ${EXPORTED_MODEL_PATH}/decoder.xml decoder.xml
COPY --chown=user:user ${EXPORTED_MODEL_PATH}/decoder.bin decoder.bin
