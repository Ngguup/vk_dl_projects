import numpy as np

class LinearRegressorGD:
    """
    Линейная регрессия с использованием Gradient Descent
    """

    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        Конструктор класса

        Параметры:
            learning_rate (float): Скорость обучения
            n_iter (int): Количество итераций градиентного спуска
        """
        np.random.seed(42)

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        pass

    def get_loss_grad(self, X, y):
        return (2 / X.shape[0]) * X.T @ (self.predict(X) - y)

    def fit(self, X, y):
        """
        Обучение модели на обучающей выборке с использованием
        градиентного спуска

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)
            y (np.ndarray): Вектор таргета длины n_samples
        """
        y = y.ravel()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            self.w -= self.learning_rate * self.get_loss_grad(X, y)
        return self

    def predict(self, X):
        """
        Получение предсказаний обученной модели

        Параметры:
            X (np.ndarray): Матрица признаков

        Возвращает:
            np.ndarray: Предсказание для каждого элемента из X
        """
        if X.shape[1] != self.w.shape[0]:
            return np.hstack((np.ones((X.shape[0], 1)), X)) @ self.w
        return X @ self.w

    def get_params(self):
        """
        Возвращает обученные параметры модели
        """
        return self.w
