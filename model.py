import tensorflow as tf 
import numpy as np 
from polyron_layer import Polyron
from tensorflow.keras.layers import Dense, Flatten

log_path = "/Users/thomasklein/Projects/Polyron/logs/"
archive_path = "/Users/thomasklein/Projects/Polyron/archive/"
name = "first_test"

batchsize = 32

def main():

    # create fashion mnist dataset 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # create model
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Polyron(128, degree=5, initializer=tf.keras.initializers.TruncatedNormal()))
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # create callbacks for training
    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir=log_path+name),
        tf.keras.callbacks.ModelCheckpoint(filepath=archive_path+name+".h5",
                                            save_best_only=False,
                                            period=1)
    ]

    model.fit(x_train, y_train,
        epochs=15,
        steps_per_epoch=100,
        validation_data=(x_test, y_test),
        validation_steps=10_000//batchsize,
        callbacks = callbacks)

if __name__ == "__main__":
    main()