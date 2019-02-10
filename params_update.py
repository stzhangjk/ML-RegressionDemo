import math


class ParametersGenerator:
    g_w_list = []
    g_b_list = []

    def __init__(self, w_init, b_init, lr_init, training_data):
        """

        :param w_init:
        :param b_init:
        :param lr_init:
        :param training_data:
        """
        self.w = w_init
        self.b = b_init
        self.lr_w = lr_init
        self.lr_b = lr_init
        self.training_data = training_data

    def g_w(self):
        return 2 / len(self.training_data) * sum(map(
            lambda data: self.w * data['x']**2 + self.b * data['x'] - data['x'] * data['y'], self.training_data))

    def g_b(self):
        return 2 / len(self.training_data) * sum(map(
            lambda data: self.w * data['x'] + self.b - data['y'], self.training_data))

    def __iter__(self):
        return self

    def __next__(self):
        next_g_w = self.g_w()
        next_g_b = self.g_b()
        self.g_w_list.append(next_g_w)
        self.g_b_list.append(next_g_b)
        self.w -= self.lr_w / math.sqrt(sum(map(lambda x: x * x, self.g_w_list))) * next_g_w
        self.b -= self.lr_b / math.sqrt(sum(map(lambda x: x * x, self.g_b_list))) * next_g_b
        return self.w, self.b
