import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import argparse
import pickle
import matplotlib.pyplot as plt 
import numpy as np
from model import create_model

parser = argparse.ArgumentParser()
parser.add_argument('-dl', '--download', action='store_true')

args = parser.parse_args()
if args.download:
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    with open('data/x_train.pickle','wb') as f:
        pickle.dump(x_train, f)
        f.close()
    with open('data/y_train.pickle','wb') as f:
        pickle.dump(y_train, f)
        f.close()
    with open('data/x_test.pickle','wb') as f:
        pickle.dump(x_test, f)
        f.close()
    with open('data/y_test.pickle','wb') as f:
        pickle.dump(y_test, f)
        f.close()
else:
    with open('data/x_train.pickle','rb') as f:
        x_train = pickle.load(f)
        f.close()
    with open('data/y_train.pickle','rb') as f:
        y_train = pickle.load(f)
        f.close()
    with open('data/x_test.pickle','rb') as f:
        x_test = pickle.load(f)
        f.close()
    with open('data/y_test.pickle','rb') as f:
        y_test = pickle.load(f)
        f.close()


# rescale images from 0-255.0 to 0-1.0
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
model = create_model()
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1, validation_data=(x_test, y_test))
cnn_results = model.evaluate(x_test, y_test)
print(cnn_results)

model.save('model/digit_rec.h5', save_format="h5")