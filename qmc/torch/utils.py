import torch
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def get_moons():
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]
    return (X_train, y_train), (X_test, y_test)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]