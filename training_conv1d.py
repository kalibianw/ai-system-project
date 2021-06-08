from utils import DataModule, CNN1D

from tensorflow.keras.utils import to_categorical
from torchinfo import summary
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE}")

nploader = np.load("npz/TFSR_n_mfcc_80.npz")
x_norm_data, y_data = np.swapaxes(nploader["x_norm_data"], 1, 2), to_categorical(nploader["y_data"])
print(np.shape(x_norm_data), np.shape(y_data))

model = CNN1D(32, classes=30).to(DEVICE)
summary(model, input_size=(32, 32, 80))
