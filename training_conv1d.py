from utils import DataModule, TrainModule

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import shutil
import os


CKPT_PATH = "D:/AI/ckpt/ai-system-project/training_conv1d.ckpt"
MODEL_PATH = "D:/AI/model/ai-system-project/training_conv1d.h5"
LOG_FOLDER_PATH = "./logs/training_conv1d/"
if os.path.exists(LOG_FOLDER_PATH):
    shutil.rmtree(LOG_FOLDER_PATH)
    os.makedirs(LOG_FOLDER_PATH)
EPOCHS = 100
REDUCE_LR_RATE = 0.6
LOG_INTERVAL = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 256

dm = DataModule(None, batch_size=BATCH_SIZE)
nploader = np.load("npz/TFSR_n_mfcc_80.npz")
x_norm_data, y_data = nploader["x_norm_data"], to_categorical(nploader["y_data"])
x_train_all, x_test, y_train_all, y_test = train_test_split(x_norm_data, y_data, test_size=0.4)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.2)
print(np.shape(x_train), np.shape(x_valid), np.shape(x_test),
      np.shape(y_train), np.shape(y_valid), np.shape(y_test))

tm = TrainModule(
    input_shape=np.shape(x_train[0, :, :]),
    classes=np.shape(y_train)[1],
    batch_size=BATCH_SIZE
)

model = tm.conv1d_model()
model.summary()

hist = tm.model_training(
    model,
    x_train, y_train, x_valid, y_valid,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    reduce_lr_rate=REDUCE_LR_RATE,
    log_dir_path=LOG_FOLDER_PATH,
    ckpt_path=CKPT_PATH,
    model_path=MODEL_PATH
)
