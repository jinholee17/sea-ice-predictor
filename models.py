import tensorflow as tf

class LSTMModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LSTMModel, self).__init__()
        self.layers = tf.keras.layers.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape), 
            tf.keras.layers.Dense(1)])

    def call(self, inputs):
        logits = self.layers(inputs)
        return logits 
    
class CNNModel(tf.keras.model): 
    def __init__(self, input_shape): 
        super(CNNModel, self).__init__()
        self.layers = tf.layers.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape = input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.LSTM(50, activation = 'relu'),
            tf.keras.layers.Dense(1)
        ])
    def call(self, inputs): 
        logits = self.layers(inputs)
        return logits