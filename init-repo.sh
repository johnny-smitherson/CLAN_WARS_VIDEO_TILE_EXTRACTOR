#!/bin/bash
set -ex


ffmpeg -version || ( echo "ffmpeg not found" && exit 1 )
magick --version || ( echo "magick not found" && exit 1 )
python --version || ( echo "python not found" && exit 1 )

if ! [ -d ".venv" ]; then
    python -m venv .venv
fi
. .venv/Scripts/activate

mkdir -p downloads/tracks



pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 typing_extensions==4.12.2


pip install  ultralytics

pip install opencv-contrib-python
