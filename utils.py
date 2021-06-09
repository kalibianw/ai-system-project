# For preprocessing
from pydub.utils import mediainfo
import numpy as np
import librosa
import os

# For training
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

# For TensorFlow
from tensorflow.keras import models, layers, activations, losses, optimizers, callbacks


class DataModule:
    def __init__(self, audio_dir_path, batch_size):
        self.BATCH_SIZE = batch_size
        self.audio_count = 0
        self.audio_dir_path = audio_dir_path
        self.dir_list = os.listdir(self.audio_dir_path)
        self.each_audio_dir_path = list()
        if audio_dir_path is not None:
            print("audio_dir_list")
            for i in range(len(self.dir_list)):
                print(audio_dir_path + self.dir_list[i] + "/")
                self.each_audio_dir_path.append(audio_dir_path + self.dir_list[i] + "/")
                audio_list = os.listdir(self.each_audio_dir_path[i])
                self.audio_count += len(audio_list)

            print(f"total audio file: {self.audio_count}")

    def audio_mfcc(self, n_mfcc):
        features = list()
        features_norm = list()
        labels = list()
        count = 0
        per = self.audio_count / 1000

        for i in range(len(self.each_audio_dir_path)):
            audio_file_list = os.listdir(self.each_audio_dir_path[i])
            for audio_file_name in audio_file_list:
                audio_file_path = self.each_audio_dir_path[i] + audio_file_name
                audio_file, sr = librosa.load(path=audio_file_path, sr=int(mediainfo(audio_file_path)['sample_rate']))

                if np.shape(audio_file)[0] < sr:
                    expand_arr = np.zeros(shape=(sr - np.shape(audio_file)[0]), dtype="float32")
                    audio_file = np.append(audio_file, expand_arr)

                elif np.shape(audio_file)[0] > sr:
                    cutted_arr = np.split(ary=audio_file, indices_or_sections=(sr,))
                    audio_file = cutted_arr[0]

                audio_mfcc = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=n_mfcc)
                audio_mfcc_norm = librosa.util.normalize(audio_mfcc)
                features.append(audio_mfcc)
                features_norm.append(audio_mfcc_norm)
                labels.append(i)

                if (count % int(per)) == 0:
                    print(f"현재 {count}개 완료, {(count / per) / 10}%")
                count += 1

        features = np.array(features)
        features_norm = np.array(features_norm)
        labels = np.array(labels)

        return features, features_norm, labels

    def np_to_dataloader(self, x_data, y_data):
        x_tensor = torch.tensor(x_data)
        y_tensor = torch.tensor(y_data, dtype=torch.long)
        # y_tensor = F.one_hot(y_tensor)

        dataset = TensorDataset(x_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        return data_loader


class FCNN(nn.Module):
    def __init__(self, classes):
        super(FCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(
            in_features=2560,
            out_features=512
        )
        self.fcnn1 = nn.Linear(
            in_features=512,
            out_features=256,
        )
        self.fcnn2 = nn.Linear(
            in_features=256,
            out_features=128,
        )
        self.fcnn3 = nn.Linear(
            in_features=128,
            out_features=64,
        )
        self.output_layer = nn.Linear(
            in_features=64,
            out_features=classes
        )

        self.dropout = nn.Dropout()

        nn.init.kaiming_normal_(self.input_layer)
        nn.init.kaiming_normal_(self.fcnn1)
        nn.init.kaiming_normal_(self.fcnn2)
        nn.init.kaiming_normal_(self.fcnn3)
        nn.init.kaiming_normal_(self.output_layer)

    def forward(self, x):
        x = self.flatten(x)

        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fcnn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fcnn2(x)
        x = F.relu(x)

        x = self.fcnn3(x)
        x = F.relu(x)

        x = self.output_layer(x)
        x = F.softmax(x)

        return x


class CNN1D(nn.Module):
    def __init__(self, input_channels, classes):
        super(CNN1D, self).__init__()
        self.input_layer = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=5)
        self.conv1d_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.conv1d_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5)
        self.conv1d_3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5)

        self.fcnn1 = nn.Linear(in_features=5632, out_features=512)
        self.fcnn2 = nn.Linear(in_features=512, out_features=256)
        self.fcnn3 = nn.Linear(in_features=256, out_features=128)
        self.output_layer = nn.Linear(in_features=128, out_features=classes)

        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.max_pool_1d = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.max_pool_1d(x)

        x = self.conv1d_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv1d_2(x)
        x = F.relu(x)
        x = self.max_pool_1d(x)

        x = self.conv1d_3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.fcnn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fcnn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fcnn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        x = F.softmax(x)

        return x


class CNN2D(nn.Module):
    def __init__(self, input_channels, classes):
        super(CNN2D, self).__init__()
        self.input_layer = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3))
        self.conv2d_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))

        self.fcnn1 = nn.Linear(in_features=256 * 15 * 3, out_features=512)
        self.fcnn2 = nn.Linear(in_features=512, out_features=256)
        self.fcnn3 = nn.Linear(in_features=256, out_features=128)
        self.output_layer = nn.Linear(in_features=128, out_features=classes)

        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.max_pool_2d = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.max_pool_2d(x)

        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2d_2(x)
        x = F.relu(x)
        x = self.max_pool_2d(x)

        x = self.conv2d_3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.fcnn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fcnn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fcnn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        x = F.softmax(x)

        return x


class TrainModule:
    def __init__(self, device, optimizer, loss, batch_size, reduce_lr_rate):
        self.DEVICE = device
        self.BATCH_SIZE = batch_size
        self.optimizer = optimizer
        self.criterion = loss
        self.epoch = 0

        self.last_val_loss = sys.maxsize
        self.non_improve_cnt = 0

        self.REDUCE_LR_RATE = reduce_lr_rate

    def training(self, model, train_loader, valid_loader, log_interval):
        model.train()
        train_loss = 0.
        correct = 0.

        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(self.DEVICE)
            label = label.to(self.DEVICE)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{} / {}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                        self.epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        (100. * batch_idx / len(train_loader)),
                        loss.item()
                    ))

            train_loss += loss.item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        train_loss /= (len(train_loader.dataset) / self.BATCH_SIZE)
        train_accuracy = 100. * correct / len(train_loader.dataset)

        valid_acc, valid_loss = self.evaluate(model, valid_loader)
        if valid_loss > self.last_val_loss:
            if self.non_improve_cnt > 5:
                self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * self.REDUCE_LR_RATE
            self.non_improve_cnt += 1
        else:
            self.non_improve_cnt = 0

        return train_accuracy, train_loss, valid_acc, valid_loss

    def evaluate(self, model, test_loader):
        model.eval()
        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(self.DEVICE)
                label = label.to(self.DEVICE)
                output = model(image)
                test_loss += self.criterion(output, label).item()
                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= (len(test_loader.dataset) / self.BATCH_SIZE)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        return test_accuracy, test_loss


class TrainModule2:
    def __init__(self, input_shape, classes, batch_size):
        self.BATCH_SIZE = batch_size
        self.INPUT_SHAPE = input_shape
        self.output_shape = classes

    def conv2d_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.INPUT_SHAPE),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=256, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),

            layers.Flatten(),

            layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation=activations.relu, kernel_initializer="he_normal"),

            layers.Dense(self.output_shape, activation=activations.softmax)
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=['acc']
        )

        return model

    def model_training(self, model, x_train, y_train, x_valid, y_valid, epochs,
                       early_stopping_patience, reduce_lr_rate,
                       ckpt_path, log_dir_path, model_path):
        callback = [
            callbacks.EarlyStopping(patience=early_stopping_patience, verbose=1),
            callbacks.ReduceLROnPlateau(factor=reduce_lr_rate, patience=5, verbose=1, min_lr=1e-3),
            callbacks.ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True, save_weights_only=True),
            callbacks.TensorBoard(log_dir=log_dir_path)
        ]
        hist = model.fit(
            x=x_train, y=y_train,
            batch_size=self.BATCH_SIZE,
            epochs=epochs,
            verbose=1,
            callbacks=callback,
            validation_data=(x_valid, y_valid)
        )

        model.load_weights(filepath=ckpt_path)
        model.save(filepath=model_path)

        return hist
