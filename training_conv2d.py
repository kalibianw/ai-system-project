from utils import DataModule, CNN2D, TrainModule

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np
import shutil
import torch
import sys
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE}")
MODEL_PATH = "D:/AI/model/ai-system-project/training_conv2d.pt"
LOG_FOLDER_PATH = "./logs/training_conv2d/"
if os.path.exists(LOG_FOLDER_PATH):
    shutil.rmtree(LOG_FOLDER_PATH)
    os.makedirs(LOG_FOLDER_PATH)
EPOCHS = 100
REDUCE_LR_RATE = 0.6
LOG_INTERVAL = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 32

dm = DataModule(None, batch_size=BATCH_SIZE)
nploader = np.load("npz/TFSR_n_mfcc_80.npz")
x_norm_data, y_data = np.expand_dims(nploader["x_norm_data"], axis=1), nploader["y_data"]
print(np.shape(x_norm_data), np.shape(y_data))

x_train_all, x_test, y_train_all, y_test = train_test_split(x_norm_data, y_data, test_size=0.4)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.2)
print(np.shape(x_train), np.shape(x_valid), np.shape(x_test),
      np.shape(y_train), np.shape(y_valid), np.shape(y_test))

train_loader = dm.np_to_dataloader(x_train, y_train)
valid_loader = dm.np_to_dataloader(x_valid, y_valid)
test_loader = dm.np_to_dataloader(x_test, y_test)

model = CNN2D(1, classes=30).to(DEVICE)
summary(model, input_size=(32, 1, 80, 32))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.CrossEntropyLoss()
tm = TrainModule(
    device=DEVICE,
    optimizer=optimizer,
    loss=loss,
    batch_size=BATCH_SIZE,
    reduce_lr_rate=REDUCE_LR_RATE
)

writer = SummaryWriter(log_dir=LOG_FOLDER_PATH)
best_loss = sys.maxsize
not_improve_cnt = 0
for Epoch in range(1, EPOCHS + 1):
    train_acc, train_loss, valid_acc, valid_loss = tm.training(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        log_interval=LOG_INTERVAL
    )
    print("\n[EPOCH: {}], \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f}%".format(Epoch, train_loss, train_acc))
    print("[EPOCH: {}], \tValid Loss: {:.4f}, \tValid Accuracy: {:.2f}%\n".format(Epoch, valid_loss, valid_acc))

    test_acc, test_loss = tm.evaluate(model, test_loader)
    print("[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}%\n".format(Epoch, test_loss, test_acc))

    writer.add_scalar("Loss/train", train_loss, Epoch)
    writer.add_scalar("Loss/valid", valid_loss, Epoch)
    writer.add_scalar("Loss/test", test_loss, Epoch)
    writer.add_scalar("Accuracy/train", train_acc, Epoch)
    writer.add_scalar("Accuracy/valid", valid_acc, Epoch)
    writer.add_scalar("Accuracy/test", test_acc, Epoch)

    if test_loss < best_loss:
        torch.save(model.state_dict(), MODEL_PATH)
        not_improve_cnt = 0
    if test_loss > best_loss:
        if not_improve_cnt > EARLY_STOPPING_PATIENCE:
            break
        not_improve_cnt += 1
