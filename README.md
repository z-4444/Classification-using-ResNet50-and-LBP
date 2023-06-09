# Shoe, Sandal, and Boot Classification using ResNet50 and Local Binary Patterns (LBP)

# Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Usage](#usage)
7. [Requirements](#requirements)
8. [Docker](#docker)

## Introduction
This model uses ResNet50 and Local Binary Patterns (LBP) to classify images of shoes, sandals, and boots. The model is served via a Flask web server that can be run using a flask API. Additionally, a Docker image has been created to package the entire application. The README.md file for the project contains a table of contents that outlines the contents of the file.

## Dataset
The dataset used for this model is the Shoe vs Sandal vs Boot Dataset, which contains 15000 images of shoes, sandals, and boots. The images are divided into three directories: Shoe, Sandal, and Boot, each containing 5000 images.

## Model Architecture
The model architecture consists of two feature extraction methods: ResNet50 and LBP. ResNet50 is a pre-trained convolutional neural network that can extract high-level features from images, while LBP is a texture descriptor that can extract local patterns from images. The ResNet50 model is used to extract global features from the images, while the LBP descriptor is used to extract local patterns from the images.

The ResNet50 model is loaded with pre-trained weights from ImageNet, and the last layer (the fully connected layer) is removed. The output of the ResNet50 model is then fed into a concatenation layer along with the LBP features and the height and width of the image. The output of the concatenation layer is then fed into a support vector machine (SVM) classifier for training and testing.
## Training
The model is trained on 12000 images and tested on 3000 images. The images are randomly split into training and testing sets using a 80:20 ratio. The features extracted from the images are normalized using StandardScaler before training the SVM classifier. The SVM classifier is trained using an RBF kernel with a regularization parameter of C=10 and a gamma value of 'auto'.

## Evaluation Metrics
The model is evaluated using accuracy, precision, recall, and F1 score metrics. <br />
Accuracy:    0.99 <br />
Precision:   0.99 <br />
Recall:      0.99  <br />
F1 score:    0.99 <br />

The confusion matrix is also generated to visualize the performance of the model.

![confusion matrix](https://user-images.githubusercontent.com/105338831/229350287-75194225-8fa2-407c-86c6-0e77e245afbd.png)


## Usage
To run the model, follow these steps:

* Download the Shoe vs Sandal vs Boot Dataset.
* Clone the repository.
* Change the shoe_dir, sandal_dir, and boot_dir variables in the code to the path of the Shoe, Sandal, and Boot directories, respectively.
* Run the code to extract the features from the images and save them in a CSV file.
* Run the code to train and test the SVM classifier on the extracted features.

## Requirements
The following Python packages are required to run the model:

* pandas
* scikit-learn
* scikit-image
* OpenCV
* NumPy
* TensorFlow

## Docker
You can pull my docker image from [here](https://hub.docker.com/repository/docker/mzahid4444/fruit_classify/general).
To pull run following commond.

 ``` docker push mzahid4444/fruit_classify:tagname  ```
> **Note**
> Note that the image name is fruit_classify but it is actually for a shoe classifier.

Following are some commonds that are for built, delete, run and save the image. <br />

To build the Docker image, navigate to the directory containing the Dockerfile and run the following command: <br />
``` sudo docker build -t footwear-classifier . ``` <br />
To delete the Docker image, use the following command: <br />
``` sudo docker rmi --force footwear-classifier ``` <br />
To run the Docker image, use the following command: <br />
``` sudo docker run -p 5000:5000 footwear-classifier ``` <br />
This command maps port 5000 on your local machine to port 5000 on the Docker container. <br />

To save the Docker image to a tar file, use the following command: <br />
``` sudo docker save ImageId -o fName.tar ``` <br />
Replace ImageId with the ID of the Docker image you want to save and fName.tar with the desired name for the tar file.

