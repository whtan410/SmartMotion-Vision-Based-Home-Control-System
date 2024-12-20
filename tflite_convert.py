import tensorflow
from tensorflow.keras.models import load_model

model = tensorflow.keras.models.load_model(r'C:\Users\pa662\PycharmProjects\HGM\V11\gestpred.h5')
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("gestpred.tflite", "wb").write(tflite_model)


