#pip install tensorflow --user
#pip install scikit.learn --user
import pickle
import os
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
#from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import backend

# Load dataset
data_directory = "data/"
x_train = np.load(os.path.join(data_directory, "x_train256.npy"))
y_train = np.load(os.path.join(data_directory, "y_train256.npy"))
y_train = pd.DataFrame(y_train.astype(int))

x_valid = np.load(os.path.join(data_directory, "x_valid256.npy"))
y_valid = np.load(os.path.join(data_directory, "y_valid256.npy"))
y_valid = pd.DataFrame(y_valid.astype(int))

x_test = np.load(os.path.join(data_directory, "x_test256.npy"))
y_test = np.load(os.path.join(data_directory, "y_test256.npy"))
y_test = pd.DataFrame(y_test.astype(int))   

def create_model():
    # Convolutional Neural Network
    model = Sequential()
    #Input Shape: width: 256 x height: 256 x depth: 3 (color channels: RGB)
    #Convolutional: filters: 32, kernel size: 6x6
    #Activation: ReLu: "rectified linear unit", defined as y=max(0,x)
    model.add(Conv2D(32, (6, 6), input_shape=(256, 256, 3), activation='relu'))
    #Pooling: MaxPooling of size: 3x3
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (6, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (6, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #Flatten: to transform a two-dimensional matrix into a vector
    model.add(Flatten())
    #Dense: used to change the dimensions of the vector
    model.add(Dense(500, activation='relu'))
    #Dropout: to reduce overfitting
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(4))
    #Adam = Adaptive moment estimation, Measure: MAE
    model.compile(optimizer='Adam', loss='mean_absolute_error')
    #Epochs: times updating
    #Batch size: update weights after 128 images
    return model

def train_model():
    model = create_model()
    # Without Data Augmentation
    # model.fit(x_train, y_train, epochs=40, batch_size=128)
    # pickle.dump(model, open("model.pkl", "wb"))

    #Data Augmantation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8,1.2),
        horizontal_flip=True,
        channel_shift_range=50.0, 
        fill_mode='nearest',
        validation_split=0.2
    )
    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size = 128), 
                        validation_data = (x_valid, y_valid), 
                        epochs = 40)
    pickle.dump(model, open("model_aug.pkl", "wb"))
    return model
    
def predict():
    model = train_model()
    prediction = model.predict(x_test)
    prediction[prediction < 1] = 1
    prediction[prediction > 5] = 5  
    # calculate several metrics
    maecnn = mean_absolute_error(y_test, prediction)
    msecnn = mean_squared_error(y_test, prediction)
    # get some info
    print(prediction)
    print("several performance metrics CNN:")
    print("mae:", maecnn) 
    print("mse:", msecnn)
    model.summary()

predict()
