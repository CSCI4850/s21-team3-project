import tensorflow as tf
import numpy as np
import pandas as pd

loaded = tf.saved_model.load("Model/model_pb") # Load Model
model = loaded.signatures["serving_default"]   # Load with default signature

data = tf.convert_to_tensor(np.random.rand(1, 1960), dtype=np.float32) # Grab input Data - Create Random numbers of input shape 

print(np.argmax(model(data)["predictions"]))      # Run Model Prediction
