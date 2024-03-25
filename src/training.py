from tensorflow import keras
from tensorflow.keras import models,layers

def training(train_data, train_lbl):
    cnn1=models.Sequential()
    cnn1.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    cnn1.add(layers.MaxPooling2D((2,2)))
    cnn1.add(layers.Conv2D(32,(3,3),activation='relu'))
    cnn1.add(layers.MaxPooling2D((2,2)))
    cnn1.add(layers.Conv2D(32,(3,3),activation='relu'))

    cnn1.add(layers.Flatten())
    cnn1.add(layers.Dense(64,activation='relu'))
    cnn1.add(layers.Dense(64,activation='relu'))
    cnn1.add(layers.Dense(6,activation='softmax'))
    cnn1.summary()
    cnn1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    trained = cnn1.fit(train_data,train_lbl,epochs=50,validation_split=0.2,callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    return trained