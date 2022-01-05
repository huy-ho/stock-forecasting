import keras
import tensorflow as tf


model = tf.keras.models.load_model('keras_model.h5')

prediction = model.predict