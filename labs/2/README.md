# Нейронные сети

1. Постройте нейронную сеть из одного нейрона и обучите её на датасетах nn_0.csv и nn_1.csv. Насколько отличается результат обучения и почему? Сколько потребовалось эпох для обучения? Попробуйте различные функции активации и оптимизаторы. 
2. Постройте другую нейронную сеть, чтобы достичь минимальной ошибки на датасете nn_1.csv. Объясните выбор гиперпараметров построенной нейронной сети.
3. Создайте классификатор на базе свёрточной нейронной сети для набора данных MNIST (так же можно загрузить с помощью torchvision.datasets.MNIST, tensorflow.keras.datasets.mnist.load_data и пр.). Приведите архитектуру полученной свёрточной нейронной сети (количество свёрточных слоёв с размерами, какой был пулинг и т.д.). Оцените качество классификации. Визуализируйте фильтры (filters) и результирующие признаки (feature maps) (Пример визуализации ).
