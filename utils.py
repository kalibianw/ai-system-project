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


class DataModule:
    def __init__(self, audio_dir_path, batch_size):
        self.BATCH_SIZE = batch_size
        self.audio_count = 0
        self.audio_dir_path = audio_dir_path
        self.dir_list = os.listdir(self.audio_dir_path)
        self.each_audio_dir_path = list()
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
        x_tensor = torch.Tensor(x_data)
        y_tensor = torch.Tensor(y_data)

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
