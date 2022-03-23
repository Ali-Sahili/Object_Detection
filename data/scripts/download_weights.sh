#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov5/releases
# Usage:
#    $ bash path/to/download_weights.sh

python3 - <<EOF
import torch

assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
          'yolov5s6.pt','yolov5m6.pt','yolov5l6.pt','yolov5x6.pt']
filename = './'
for name in assets:
    url=f'https://github.com/ultralytics/yolov5/releases/download/v5.0/{name}'
    torch.hub.download_url_to_file(url, filename)

EOF
