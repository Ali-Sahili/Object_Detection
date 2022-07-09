![Object Detection](https://img.shields.io/static/v1?style=plastic&message=Object+Detection&color=013243&logo=Object+Detection&logoColor=FFFFFF&label=)
![YOLO v5](https://img.shields.io/static/v1?style=plastic&message=YOLO&color=222222&logo=YOLO&logoColor=00FFFF&label=)

# Object_Detection
Implementation of YOLO-v5 algorithm to detect multiple objects on both images and streaming

## Requirements
To install all necessary libraries:
`pip3 install -r requirements.txt`

## Usage
* To install weights:

`bash data/scripts/download_weights.sh`

* To get CoCo dataset:

`bash data/scripts/get_coco.sh`

* For training:

`python3 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --epochs 300`

* For testing:

`python3 test.py --data coco128.yaml --weights yolov5s.pt --img 640`

* To detect objects in a specified images given model's weights:

`python3 detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640`

* To play with videos: (e.g. webcam):

`python3 detect_video.py --source 0 --weights yolov5s.pt --img 640`

### Acknowledgement
This code is a modified version from [this repository](https://github.com/ultralytics/yolov5/).
