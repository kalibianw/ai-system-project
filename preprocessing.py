from utils import DataModule
import numpy as np

DATASET_PATH = "D:/AI/data/TFSR_train/audio/"

dm = DataModule(audio_dir_path=DATASET_PATH)

x_data, x_norm_data, y_data = dm.audio_mfcc(n_mfcc=80)
print(np.shape(x_data), np.shape(x_norm_data), np.shape(y_data))
np.savez_compressed("npz/TFSR_n_mfcc_80.npz", x_data=x_data, x_norm_data=x_norm_data, y_data=y_data)
