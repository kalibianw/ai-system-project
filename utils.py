# For preprocessing
from pydub.utils import mediainfo
import numpy as np
import librosa
import os

# For training
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


class TrainModule:
    def __init__(self, input_shape, classes, batch_size):
        self.BATCH_SIZE = batch_size
        self.INPUT_SHAPE = input_shape
        self.output_shape = classes

    def fully_connected_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.INPUT_SHAPE),
            layers.Flatten(),

            layers.Dense(units=1024, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(units=512, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(units=256, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.Dense(units=128, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dense(units=self.output_shape, activation=activations.relu, kernel_initializer="he_normal"),
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def conv1d_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.INPUT_SHAPE),
            layers.Conv1D(filters=128, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.MaxPooling1D(),

            layers.Conv1D(filters=256, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),

            layers.Conv1D(filters=256, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal"),
            layers.MaxPooling1D(),

            layers.Conv1D(filters=512, kernel_size=5, activation=activations.relu, kernel_initializer="he_normal"),
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
