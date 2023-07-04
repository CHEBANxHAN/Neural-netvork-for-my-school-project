import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time

# скачиваем данные и разделяем на надор для обучения и тесовый
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#ормализация данных
x_train = x_train / 255
x_test = x_test / 255

#Преобразование входных значений в векторы по категориям
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

#Формирование НС
model = keras.Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)), #Число ядер, размер ядер, сохранение размера карты, функция активации, входы
    MaxPooling2D((2, 2), strides=2), #Уменьшение маштаба, выделяя квадрат пикселей 2x2, выбирая самый большой пиксель и смещая квадрат на два пикселя
    Conv2D(64, (3, 3), padding="same", activation="relu"), #Число ядер, размер ядер, сохранение размера карты, функция активации
    MaxPooling2D((2, 2), strides=2), #Уменьшение маштаба, выделяя квадрат пикселей 2x2, выбирая самый большой пиксель и смещая квадрат на два пикселя
    Flatten(), #Формирование входного слоя для полноценной НС
    Dense(128, activation="relu"), #Формирование скрытого слоя для полноценной НС
    Dense(10, activation="softmax") #Формирование выходного слоя для полноценной НС
])


optim = keras.optimizers.AdamW()

#Оптимизация по Adam с критерием качества - категориальная кросс-энтропия
model.compile(optimizer=optim,
              loss="categorical_crossentropy",
              metrics=["accuracy"]) #Уменьшение процента ошибки

#Обучение: 80% - обучающая выборка и 20% - выборка валидации
his = model.fit(x_train, y_train_cat, epochs=100, validation_split=0.2, verbose=False)
print("Обучено")

model.save("learn.h5")
print("Сохранено")
