#Converts .h5 model to .tflite for mobile use.
#You’ll integrate this .tflite file into Flutter or React Native using TensorFlow Lite APIs.

import tensorflow as tf

model = tf.keras.models.load_model("models/regular_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("models/yoga_pose_model.tflite", "wb") as f:
    f.write(tflite_model)
