import Neural_Network
import numpy as np
import matplotlib.pyplot

#Количество узлов между слоями
ind = 784
hnd = 200
ond = 10

#Коэффициент обучения
lr = 0.2
#Создание объекта
n = Neural_Network.neuralNetwork(ind, hnd, ond, lr)

#Загрузка списка
train_data_file = open("C:/Users/vikto/Neural Network/mnist_dataset/mnist_train_100.csv", "r")
train_data_list = train_data_file.readlines()
train_data_file.close()

#Тренировка нейронных сетей с эпохами
epochs = 5

for e in range(epochs):
    #Создание тренировочного набора
    for record in train_data_list:
        #Список с данными из файла
        all_value = record.split(",")
        #Изменение значений списка в зависимости от сигмоиды
        inputs = (np.asfarray(all_value[1:]) / 255.0 * 0.99) + 0.01
        #Создание списка для маркеров
        targets = np.zeros(ond) + 0.01
        #Принимаем целевое значение 0.99 для данной записи
        targets[int(all_value[0])] = 0.99
        #Обучение нейронной сети
        n.train(inputs, targets)
    pass
        

#Журнал оценок работы сети
scorecard = []

#Тестирование нейронной сети

test_data_file = open("C:/Users/vikto/Neural Network/mnist_dataset/mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

#Создание тестого набора данных
for records in test_data_list:
    #Список с данными из файла
    all_values = records.split(",")
    true_ans = int(all_values[0])
    print(f"{true_ans} -- истинный маркер")
    #Изменение значений списка в зависимости от сигмоиды
    scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #Опрос сети
    out = n.query(scaled_input)
    #Получение ответа сети
    lb = np.argmax(out)
    print(f"{lb} -- ответ сети")
    #Оценка ответа сети
    if lb == true_ans:
        scorecard.append(1)
    else:
        scorecard.append(0)

    image_array = np.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()
    pass

print(scorecard)

#Расчёт эффективности нейронной сети
score_data = np.asarray(scorecard)
print(f"Эффективность - {score_data.sum() / score_data.size * 100} %")
