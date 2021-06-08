from utils import DataModule, CNN2D

from tensorflow.keras.utils import to_categorical
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE}")

nploader = np.load("npz/TFSR_n_mfcc_80.npz")
x_norm_data, y_data = np.expand_dims(nploader["x_norm_data"], axis=1), to_categorical(nploader["y_data"])
print(np.shape(x_norm_data), np.shape(y_data))

model = CNN2D(1, classes=30).to(DEVICE)
summary(model, input_size=(32, 1, 80, 32))
