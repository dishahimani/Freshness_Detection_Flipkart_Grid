# Flipkart Grid Competition: Object Detection, Classification, and Freshness Detection

Overview

This repository contains the code and models for our submission to the Flipkart Grid competition. Our solution focuses on two key features:

## 1. Object Detection, Classification, and Counting:
Object Detection: Accurately identifies and locates objects within images, including fruits, vegetables, and other grocery items.
Object Classification: Categorizes detected objects into specific classes (e.g., apple, banana, tomato) and brands.
Object Counting: Determines the quantity of each object category present in the image.

## 2. Freshness Detection:
Freshness Assessment: Evaluates the freshness level of fruits and vegetables using image analysis techniques.
Variety Handling: Accommodates different types of fruits and vegetables, considering factors like color, texture, and shape.
Model Architecture

## Working
Our solution leverages a combination of state-of-the-art machine learning and deep learning techniques:
YOLOv8: A powerful object detection model for accurate and efficient object localization and classification.
Transfer Learning: Utilizes pre-trained models on large datasets to improve performance and reduce training time.
Convolutional Neural Networks (CNNs): Extracts relevant features from images, such as color, texture, and shape.
Transfer Learning: Leverages pre-trained models like ResNet or EfficientNet for feature extraction.
Custom Classifier: Trains a classifier to predict freshness levels based on extracted features.
Streamlit: A user-friendly Python library for building web applications.
