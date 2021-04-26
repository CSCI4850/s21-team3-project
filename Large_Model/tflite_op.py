import tensorflow as tf
import numpy as np
from feature_extract import run_Micro_process


def save_float_model(model_path, save_path):
    """Convert Model into a Quantized TFLITE Model
    
        Args:
            model_path: path to the saved .pb Model
            save_path: path to save tflite Model
        returns:
            None
    """
    with tf.compat.v1.Session() as sess:        
        float_converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        float_tflite_model = float_converter.convert()
        float_tflite_model_size = open(save_path, "wb").write(float_tflite_model)
        print("Float model is %d bytes" % float_tflite_model_size)

def save_quantized_model(model_path, save_path, rep_files, rep_data=100):
    
    """ Convert Model into a Quantized TFLITE Model
    
        Args:
            model_path: path to the saved .pb Model
            save_path: path to save tflite Model
            test_files: path to pull reperesentative data from
            rep_data: Number of reperesentative data defaulted to 100
        returns:
            None
    """

    
    with tf.compat.v1.Session() as sess:  
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.compat.v1.lite.constants.INT8  
        converter.inference_output_type = tf.compat.v1.lite.constants.INT8 
        def representative_dataset_gen():
            for i in range(rep_data):
                data = run_Micro_process(rep_files[i], sess)
                flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
                yield [flattened_data]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        tflite_model_size = open(save_path, "wb").write(tflite_model)
        print("Quantized model is %d bytes" % tflite_model_size)


def predict_float(data, float_model_path):
    """ Predict an input data from a converted float Model
        Args:
            data: vector to predict class
            float_model_path: path to float tflite model
        return:
            predicted class
    """
    
    data = np.expand_dims(data, axis=0).astype(np.float32)
    assert data.shape[1] == 1960, "Shape should be 1960"
    
    interpreter = tf.lite.Interpreter(float_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    return output.argmax()


def predict_quantized(data, Quantized_model_path):
    """ Predict an input data from a converted Quantized-int8 Model
    
        Args:
            data: vector to predict class
            float_model_path: path to float tflite model
        return:
            predicted class

    """
    data = np.expand_dims(data, axis=0).astype(np.float32)
    assert data.shape[1] == 1960, "Shape should be 1960 vector length"
    
    interpreter = tf.lite.Interpreter(Quantized_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    data = data / input_scale + input_zero_point
    data = data.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    return output.argmax()

