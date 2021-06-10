from utils import TrainModule

import numpy as np
import shutil
import os

CKPT_PATH = "D:/AI/ckpt/ai-system-project/training_conv2d.ckpt"
MODEL_PATH = "D:/AI/model/ai-system-project/training_conv2d.h5"
LOG_FOLDER_PATH = "./logs/training_conv2d/"
if os.path.exists(LOG_FOLDER_PATH):
    shutil.rmtree(LOG_FOLDER_PATH)
    os.makedirs(LOG_FOLDER_PATH)
EPOCHS = 1000
REDUCE_LR_RATE = 0.6
LOG_INTERVAL = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 256

nploader = np.load("npz/TFSR_n_mfcc_80_splited.npz")
x_train, x_valid, x_test, y_train, y_valid, y_test = \
    np.expand_dims(nploader["x_train"], axis=-1), \
    np.expand_dims(nploader["x_valid"], axis=-1), \
    np.expand_dims(nploader["x_test"], axis=-1), \
    nploader["y_train"], nploader["y_valid"], nploader["y_test"]
print(np.shape(x_train), np.shape(x_valid), np.shape(x_test),
      np.shape(y_train), np.shape(y_valid), np.shape(y_test))

tm = TrainModule(
    input_shape=np.shape(x_train[0, :, :, :]),
    classes=np.shape(y_train)[1],
    batch_size=BATCH_SIZE
)

model = tm.conv2d_model()
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
