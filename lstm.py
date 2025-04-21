import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LSTMModel, self).__init__()
        self.layers = tf.keras.layers.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape), 
            tf.keras.layers.Dense(1)])

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)