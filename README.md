
# Pyro-Smoke Guard: A Comprehensive Fire and Smoke Detection System

Welcome to Pyro-Smoke Guard, a unified solution that combines two powerful object detection systems for enhanced fire and smoke detection: the YOLO Object Detection System and Pyro-Smoke Guard. This repository leverages multiple YOLO models, including YOLOv5, YOLOv8, and YOLOX, alongside a Convolutional Neural Network (CNN) based on the AlexNet architecture. Together, these components create a robust and versatile system capable of accurate detection and real-time alerting.

### Pyro-Smoke Guard
Pyro-Smoke Guard employs a CNN based on the AlexNet architecture to classify images into three categories: Fire, Neutral, and Smoke. The system is trained on a dataset comprising images of fire, smoke, and neutral scenes.

#### Datasets Used
[Forest Fire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset
)

[Forest Fire](https://www.kaggle.com/datasets/kutaykutlu/forest-fire
)

The system consists of the following key components:

### Model Training:
 The AlexNet-based model is trained on a dataset of annotated images using PyTorch. The training process includes data augmentation techniques such as random rotation, random horizontal flip, and random cropping.

### Model Evaluation:
 The trained model's performance is evaluated on a separate test dataset to assess its accuracy and generalization capabilities.

### Inference:
 The trained model can be used for image classification. An inference script is provided to load a pre-trained model and perform inference on a given image. The script outputs the predicted class and confidence score.

### Image Preprocessing:
 Image preprocessing scripts demonstrate denoising, contrast stretching, selective colorization, and color filtering techniques to enhance image features relevant to fire and smoke detection.

### Telegram Integration:
 The system includes integration with Telegram to provide real-time alerts when fire or smoke is detected. The Telegram bot sends an alert message along with the processed image.

 
## Usage
#### Training
To train the model, run the train_model.py script. Specify the paths to the training and testing datasets, and the number of epochs.

```bash
  python train_model.py --train_data_dir './Datasets/Training_data' --test_data_dir './Datasets/Test_data' --n_epochs 20

```

#### Inference

```bash
python inference_script.py --img_path 'path/to/your/testImage.jpg'

```

#### Telegram Integration
 To enable Telegram integration, provide your Telegram bot token and chat ID in the 'web/app.py' script.

 ```bash
 bot_token = 'YOUR_TELEGRAM_BOT_TOKEN'
target_chat_id = 'YOUR_TELEGRAM_CHAT_ID'
```
Run the script to send alerts to Telegram when fire or smoke is detected.

```bash
python app.py
```


## Dependencies


- [![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
- [![PyTorch](https://img.shields.io/badge/PyTorch-latest-orange.svg)](https://pytorch.org/)
- [![torchvision](https://img.shields.io/badge/torchvision-latest-orange.svg)](https://pytorch.org/vision/stable/index.html)
- [![PIL](https://img.shields.io/badge/PIL-latest-green.svg)](https://pillow.readthedocs.io/en/stable/)
- [![matplotlib](https://img.shields.io/badge/matplotlib-latest-blue.svg)](https://matplotlib.org/)
- [![NumPy](https://img.shields.io/badge/NumPy-latest-blue.svg)](https://numpy.org/)
- [![OpenCV](https://img.shields.io/badge/OpenCV-latest-blue.svg)](https://opencv.org/)
- [![telepot](https://img.shields.io/badge/telepot-latest-blue.svg)](https://telepot.readthedocs.io/en/latest/)


Install dependencies using:
```bash
pip install -r requirements.txt
```






## YOLO Object Detection System
### Overview
The YOLO Object Detection System integrates multiple YOLO models, including YOLOv5, YOLOv8, and YOLOX architectures, for smoke and fire detection in various scenarios. This system ensures adaptability and accuracy across different environments.

#### Datasets Used
[COCO128 Dataset for fire and smoke detection](https://www.kaggle.com/datasets/deeplearn1/fire-and-smoke-bbox-coco-dateset
)

[Wildfire Smoke for smoke detection](https://public.roboflow.com/object-detection/wildfire-smoke
)

[Continuous Fire for fire detection](https://universe.roboflow.com/-jwzpw/continuous_fire
)

#### Getting Started
Clone the YOLOv5 repository and install dependencies:
```bash
git clone https://github.com/ultralytics/yolov5
pip install Pillow==8.3.2
pip install -qr yolov5/requirements.txt

```
Set up data and configurations, train the models, and perform inference for smoke and fire detection.
#### YOLOv5 Smoke Detection
Train a custom YOLOv5s configuration for smoke detection:
```bash
python objectDetection/smoke-detection/yolov5/train.py --img 416 --batch 16 --epochs 100 --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results --cache
```

#### Perform smoke detection inference
```bash
python objectDetection/smoke-detection/yolov5/detect.py --weights objectDetection/smoke-detection/yolov5/runs/train/yolov5s_results/weights/best.pt --source Tests/smoke.mp4 --conf 0.4
```
#### YOLOv8 Fire Detection
Utilize YOLOv8 for fire detection, checking for the model file's existence and performing training if necessary:

#### Perform Fire detection inference
```bash
!yolo task=detect mode=predict model='fire-detectModel/best.pt' source='fire.mp4' imgsz=640 conf=0.6
```
#### YOLOX Combined Fire and Smoke Detection

Utilize the YOLOX model for combined fire and smoke detection. Train and perform inference to showcase the system's capabilities.

#### Perform Combined Fire and Smoke Detection
```bash 
python objectDetection/fire-smokeobject-detection/YOLOX/tools/demo.py video -f objectDetection/fire-smokeobject-detection/YOLOX/exps/example/custom/yolox_s.py -c {MODEL_PATH} --path {TEST_IMAGE_PATH} --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
```


### Note on Multiple YOLO Models
In the quest for achieving unparalleled excellence in object detection, our system strategically integrates a range of YOLO models, each designed to address specific challenges within diverse scenarios. The ensemble of YOLOv5, YOLOv8, and YOLOX models forms the backbone of our comprehensive object detection system. By leveraging the strengths of each individual model, our system attains adaptability, accuracy, and resilience across a spectrum of environments. Each YOLO model, standing on its own merit, contributes to the collective strength of our robust and versatile object detection framework.
### Project Structure
The project structure is organized for clarity and ease of use. Key directories include:

[Datasets](https://github.com/ayeeshaa5/Pyro-Smoke-Guard/tree/main/Datasets): This directory holds datasets specifically for training and testing the AlexNet model for image classification.

[Tests](https://github.com/ayeeshaa5/Pyro-Smoke-Guard/tree/main/Tests): Holds sample images for testing.

[Results](https://github.com/ayeeshaa5/Pyro-Smoke-Guard/tree/main/Results): Contains processed images and visualizations.

[objectDetestion](https://github.com/ayeeshaa5/Pyro-Smoke-Guard/tree/main/objectDetection): The Object Detection directory is dedicated to datasets and models pertaining to the broader field of object detection. It houses resources for advancing beyond image classification.


### Acknowledgments
- YOLOv5: [Link to YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- YOLOX: [Link to YOLOX Repository](https://github.com/Megvii-BaseDetection/YOLOX)
- YOLOv8: [Link to YOLOv8 Repository](https://github.com/ultralytics/ultralytics)
- Telegram API: [Link to Telegram API Documentation](https://core.telegram.org/api)
