import numpy as np

class OnlineScaler:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_y):
        self.n += 1
        delta = new_y - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (new_y - self.mean)

    def scale(self, y):
        if self.n == 0:
            return y
        return (y - self.mean) / np.sqrt(self.var) if self.var > 0 else y

    def restore(self, scaled_y):
        if self.n == 0:
            return scaled_y
        return scaled_y * np.sqrt(self.var) + self.mean

    @property
    def var(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 1.0
    @property
    def std(self):
        return np.sqrt(self.var)
