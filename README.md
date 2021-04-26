
# Project Description

We have built a Machine learning model as a typical smart home device that understands and performs an action upon sensing its command from a microphone. These commands include a commands to turn on and turn off home appliances such as a point of light, fan and a heater. The first steps of our machine learning workflow are quite similar to those of the traditional: we collect our data, preprocess and then design and train a model using TensorFlow. The second step is deploy our model into an embedded device. We used the Arduino NANO BLE which has a pre-installed microphone to collects an input into the model and the TFlite-Micro for Micro-controllers to run the saved Quantize - Model is saved with int8 parameters and accept int8 input - to perform an action.
The steps highlighted below will reproduce the workflow:
## Clone Repository
Open your Command Line or Terminal depending on your Operating System. You may create a virtual environment to isolate this project files from your local machine;
Read here for virtual environment :https://docs.python.org/3/library/venv.html
But for simplicity, clone this repository by runnung the command below in the terminal line by line
```
$ mkdir tiny
$ cd recommendation
$ git clone git@github.com:CSCI4850/s21-team3-project.git
```
## Intsalling Dependence
```
$ pip install -r requirements.txt
```

## Collecting Data
The data
## Preprocessing

The raw data that will be used in this project are audio signals which will be represented in a high dimensional projection for better training accuracy and faster training speed as spectrogram. We will be extracting features from our data for classification in a deep Neural Network architecture.
## Model Architecture and Training

The architecture of the Model Layers will be built on Conv2D, and MaxPool Operations from TensorFlow and SoftMax activation on the output Layer for classification of each input data. This will be possible because we will be preprocessing our input data into an image of spectrogram.
## Model Evaluation

We will train and test our Model using the preprocessed Dataset and test for an acceptable degree of precision. Performance measures for machine learning models are application-dependent. For this application, we will be testing out model based on accuracy and the inference time. These performance measures were chosen because the main functionality of our model is to recognize speech input and then quickly perform an action that will satisfy the user's need.
## Model Deployment

We will be deploying our Model into an Arduino NANO BOARD. However, embedded devices only have a very low storage memory (Usually a tens of Kilobytes), Low RAM and mostly have no operating systems. The microcontroller that will be used in this project however has an mbedOS which requires programming in C++. We will be using the TensorFlow Lite Micro to deploy our model onto an Arduino NANO micro controller and will be required to perform Quantization in order to accommodate for low memory constraint on the Microcontroller. Quantization in machine learning, means converting the model weights and Bias from a float point values to int values without influencing the accuracy of the model.
