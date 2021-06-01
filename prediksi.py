import json
import tensorflow as tf
import os

SIZE = 128
MODEL_URI = 'http://35.223.163.112:8501/v1/models/bsd:predict'

# lokasi di vmnya
model_path = "./capstone" #di sini letak model ml nya
ext_model = tf.keras.models.load_model(model_path)

def predict(data):
    prediction = ext_model.predict(data)
   # prediction = ext_model.predict(data.get("instances"))
    prediction_string = [str(pred) for pred in prediction]
    response_json = {
        "data" : data,
        #"data" : data.get("instances"),
        "prediction": list(prediction_string)
    }

    return json.dumps(response_json)