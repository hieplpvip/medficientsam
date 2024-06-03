#!/bin/bash
/usr/bin/time -f "rss=%M elapsed=%E" python3 infer.py -i /workspace/inputs/ -o /workspace/outputs/
