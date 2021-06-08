import numpy as np
import matplotlib.pyplot as plt
import librosa.display as display
from tensorflow.keras.utils import to_categorical

nploader = np.load("npz/TFSR_n_mfcc_80.npz")
x_data, x_norm_data, y_data = nploader["x_data"], nploader["x_norm_data"], to_categorical(nploader["y_data"])
print(np.shape(x_norm_data), np.shape(y_data))

for i in range(0, 5):
    display.specshow(x_data[i, :, :])
    plt.show()
