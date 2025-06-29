# 🌿 Plant Disease Classification using ResNet
This project implements a custom ResNet architecture using TensorFlow and Keras to classify plant leaf images from the PlantVillage dataset into 38 different disease categories.

## 🧠 Model Architecture
A custom-built ResNet (Residual Network) is used for classification. It includes:

Convolutional layers with Batch Normalization and ReLU

Residual blocks for deeper network training

Global Average Pooling

Fully connected (Dense) output layer with softmax activation

## 🗂 Dataset
The model uses the PlantVillage dataset which consists of 38 classes of plant diseases.

How it's loaded:
 ```!git clone https://github.com/spMohanty/PlantVillage-Dataset ```

Images are preprocessed and split using train_test_split from scikit-learn.

## 🧪 Training
Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: Customizable

Regularization: L2 applied to Conv layers

```python 
model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
## 📊 Evaluation
The model's performance is visualized using accuracy and loss curves during training and tested on a held-out validation set.

## 🖼 Sample Predictions
Random test images are shown with predicted and actual labels for visual verification.

## 📦 Requirements
TensorFlow / Keras

NumPy

OpenCV

scikit-learn

matplotlib

Create a requirements.txt with the mentioned libraries

Install all requirements:

``` bash
pip install -r requirements.txt
```
## 🚀 How to Run
Clone this repo.

Download the dataset using the provided GitHub link.

Run the notebook Resnet.ipynb.

## 📁 File Structure
``` bash
├── Resnet.ipynb          # Main notebook with model training code
├── README.md             # Project overview
└── PlantVillage-Dataset/ # Cloned dataset directory
```
