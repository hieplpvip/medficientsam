#!/bin/bash
/usr/bin/time -f "rss=%M elapsed=%E" ./main encoder.xml decoder.xml /workspace/outputs/ /workspace/inputs/ /workspace/outputs/
