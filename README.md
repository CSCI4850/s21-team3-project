
# Project Proposal

Many voice recognition software begins with keyword detection. For example, you might say "OK Google" or "Hey Siri" to wake up the corresponding recognition software on your mobile device. Once you have asked to interact, it's possible to run a model which is limited only by the resources available to the cloud provider. Most devices already send and receive audio data via the internet anyway, so it wouldn't be a bad idea to use audio interaction as a cloud-based service. However, it would be very expensive to upkeep, and introduce additional privacy risks. What if we could run this model locally on an embedded system where users are not worried about their privacy and cloud users do not necessarily have to pay a lot to upkeep their model in the cloud? We propose to build a Machine learning model as a typical smart home device that understands and performs an action upon sensing its command from a microphone. These commands will include a command to turn on and turn off devices such as a point of light, a fan e.t.c. The first steps of our machine learning workflow are quite similar to those of the traditional: we collect our data, preprocess and then design and train a model using TensorFlow. The second step is to deploy our model into an embedded device. We will be using the Arduino NANO BLE which has a pre-installed microphone to collects an input into the model and perform an action. The workflow is illustrated as below:
## Collecting Data

Dataset collection are often the most time consuming and challenging part of Machine Learning process. Since this project is expected to be delivered within a limited time-frame, we propose to use a publicly available dataset from fluent.ai and a dataset created by Pete Warden. We will also generate our own dataset which will be limited for this purpose.
## Preprocessing

The raw data that will be used in this project are audio signals which will be represented in a high dimensional projection for better training accuracy and faster training speed as spectrogram. We will be extracting features from our data for classification in a deep Neural Network architecture.
## Model Architecture and Training

The architecture of the Model Layers will be built on Conv2D, and MaxPool Operations from TensorFlow and SoftMax activation on the output Layer for classification of each input data. This will be possible because we will be preprocessing our input data into an image of spectrogram.
## Model Evaluation

We will train and test our Model using the preprocessed Dataset and test for an acceptable degree of precision. Performance measures for machine learning models are application-dependent. For this application, we will be testing out model based on accuracy and the inference time. These performance measures were chosen because the main functionality of our model is to recognize speech input and then quickly perform an action that will satisfy the user's need.
## Model Deployment

We will be deploying our Model into an Arduino NANO BOARD. However, embedded devices only have a very low storage memory (Usually a tens of Kilobytes), Low RAM and mostly have no operating systems. The microcontroller that will be used in this project however has an mbedOS which requires programming in C++. We will be using the TensorFlow Lite Micro to deploy our model onto an Arduino NANO micro controller and will be required to perform Quantization in order to accommodate for low memory constraint on the Microcontroller. Quantization in machine learning, means converting the model weights and Bias from a float point values to int values without influencing the accuracy of the model.
