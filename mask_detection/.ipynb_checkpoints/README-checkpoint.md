# Face Mask Detection using YOLO v8

## Overview
This project utilizes YOLO v8 for detecting face masks in both images and videos. The model has been trained on a dataset of approximately 6000 images, enabling it to identify faces and determine whether masks are worn or not. The detector provides two main measures: the number of individuals wearing masks and the number of individuals not wearing masks in the image. For videos, it calculates the number of scenes where masks are worn and provides a total count of such instances recorded.

## Usage
### Detecting Masks in Images or Videos
Navigate to the **Classification** folder and run the **detection.py** file:

This terminal-based interactive script will prompt for inputs to detect masks in either an image or video.



## Folder Structure
### Classification
The **Classification** folder contains scripts and files related to the face mask detection system:

- **detection.py:** A terminal-based interactive script for detecting masks in images or videos.
- **sort.py**: Script for object tracking.
- **detectiontion.ipynb**: IPython Notebook for easier understanding of the code structure and functionality.
- **requirements.txt**: File listing libraries required for the scripts to work.
- **2_class(default)**: YOLO v8 model weights for detecting face with mask and face without mask.
- **best_7**: YOLO v8 model weights optimized for detecting weights only for detecting faces with mask on.
- **Dockerfile**: Tried to containerize it but the output is around 10gb

#### yolo 
The **yolo** folder contains scripts and files related to the preprocessing and the training the model:

- **yolo**:  contains code for the preporcessing and the training process. The annotations were modiefied according to the desired ouput and changed to yolo fromat. Also all images were resized to 640 by 640 before training 
- **data.yaml:** config file for yolo to be trained
- **yolov8n.p:**pretrained weight for yolo 

#### Output 
Contains three output vedieo result for the TASK 2


## Dataset
The Dataset provided was only used to train the model.

## Model
The YOLO v8n model architecture was used for this project. The model was trained on 3461 images approximately  and 900 for validation for 30 epochs with batch size of 16. Rest of the images are used as test samples.  

## Results
The model achieved mAP50 of 0.89 and mAP50-95 of 0.61 for the two classes on the validation set. 


![Results](results.png)


