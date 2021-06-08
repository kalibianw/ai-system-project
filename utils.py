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
