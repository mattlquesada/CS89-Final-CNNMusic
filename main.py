import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import random
import tensorflow as tf
from keras import models
from keras import layers
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

TEST_LABELS = "test_labels"
TRAIN_LABELS = "train_labels"
METADATA = "musicnet_metadata.csv"
WAV_FILE_PATH = "wav_files"
FEATURE_WIDTH = 250
FEATURE_HEIGHT = 128
TEST_SPLIT = 0.75
SONG_DURATION = 5
SLICES = 6
NEPOCHS = 100


class InstrumentClassification:

    def __init__(self):

        self.df = pd.read_csv(METADATA)

        self.instruments = {}
        self.ensembles = {}

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        self.label_codes = {'Solo Piano': 0,
                            'String Quartet': 1,
                            'Accompanied Violin': 2}

    def pre_process_metadata(self):
        """
        Method for preprocessing/filtering the metadata
        :return:
        :rtype:
        """
        included_ensembles = ['Solo Piano', 'String Quartet', 'Accompanied Violin']
        self.df = self.df[self.df['ensemble'].isin(included_ensembles)]

    def get_features(self, wav_file):
        """
        Method for extracting sfft and mel spectrograms from a .wav file
        :param wav_file:
        :type wav_file:
        :return:
        :rtype:
        """

        def padding(original_array, H, W):
            A = max(0, (H - original_array.shape[0])//2)
            AA = max(0, H - A - original_array.shape[0])
            B = max(0, (W - original_array.shape[1])//2)
            BB = max(0, W - B - original_array.shape[1])

            return np.pad(original_array, pad_width=((A, AA), (B, BB)), mode='constant')

        images = []
        for i in range(SLICES):
            # Upload song segment
            offset = i*SONG_DURATION

            data, sampling_rate = librosa.load(wav_file, duration=SONG_DURATION, offset=offset)
            data, index = librosa.effects.trim(data)    # Trim preceding and trailing zeros

            # Short Time Fourier Transform and Mel Spectrogram representation
            stft = padding(np.abs(librosa.stft(data, n_fft=255, hop_length=512)), FEATURE_HEIGHT, FEATURE_WIDTH)
            MFCC = padding(librosa.feature.mfcc(data, n_fft=255, hop_length=512, n_mfcc=FEATURE_HEIGHT), FEATURE_HEIGHT, FEATURE_WIDTH)

            image = np.dstack((np.abs(stft), MFCC))
            images.append(image)

        return images

    def get_label(self, song_id):
        """
        Method for extracting the ensemble encoding of the a given .wav file
        :param song_id:
        :type song_id:
        :return:
        :rtype:
        """

        return self.df.loc[self.df['id'] == song_id, 'ensemble'].iloc[0]

    def normalize(self):
        """
        Method for normalizing data using MinMaxScaler
        :return:
        :rtype:
        """

        scaler = MinMaxScaler()

        self.X_test = np.array(self.X_test)
        self.X_train = np.array(self.X_train)

        self.X_train = scaler.fit_transform(self.X_train.reshape(-1, self.X_train.shape[-1])).reshape(self.X_train.shape)
        self.X_test = scaler.transform(self.X_test.reshape(-1, self.X_test.shape[-1])).reshape(self.X_test.shape)

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

    def create_CNN(self):
        """
        Method for creating the convolutional neural network model
        :return:
        :rtype:
        """

        input_shape = (128, FEATURE_WIDTH, 2)

        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))

        return model

    def plot(self, loss, acc, val_loss, val_acc):
        """
        Method for plotting loss and accuracy for train and validation populations
        :param loss:
        :type loss:
        :param acc:
        :type acc:
        :param val_loss:
        :type val_loss:
        :param val_acc:
        :type val_acc:
        :return:
        :rtype:
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(range(1, NEPOCHS + 1), loss, 'bo', label="training loss")
        ax1.plot(range(1, NEPOCHS + 1), val_loss, 'green', label="validation loss")

        ax1.set_title("Training Loss / Validation Loss")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(range(1, NEPOCHS + 1), acc, 'bo', label="accuracy")
        ax2.plot(range(1, NEPOCHS + 1), val_acc, 'green', label="validation accuracy")
        ax2.set_title("Training Accuracy / Validation Accuracy")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.show()

    def display_mel_spec(self, wav_file, ensemble):
        """
        Method for displaying a mel spectrogram representation of a .wav file
        :param wav_file:
        :type wav_file:
        :param ensemble:
        :type ensemble:
        :return:
        :rtype:
        """

        plt.figure(figsize=(10, 4))

        def padding(original_array, H, W):
            A = max(0, (H - original_array.shape[0]) // 2)
            AA = max(0, H - A - original_array.shape[0])
            B = max(0, (W - original_array.shape[1]) // 2)
            BB = max(0, W - B - original_array.shape[1])

            return np.pad(original_array, pad_width=((A, AA), (B, BB)), mode='constant')

        data, sampling_rate = librosa.load(wav_file, duration=SONG_DURATION, offset=0)
        data, index = librosa.effects.trim(data)  # Trim preceding and trailing zeros
        MFCC = padding(librosa.feature.mfcc(data, n_fft=255, hop_length=512, n_mfcc=FEATURE_HEIGHT), FEATURE_HEIGHT,
                       FEATURE_WIDTH)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.power_to_db(MFCC, ref=np.max), x_axis="time", y_axis="mel", fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title("Mel Spectrogram: {}".format(ensemble))

        plt.show()

    def display_stft(self, wav_file, ensemble):
        """
        Method for displaying a short time fourier transform representation of a .wav file
        :param wav_file:
        :type wav_file:
        :param ensemble:
        :type ensemble:
        :return:
        :rtype:
        """

        def padding(original_array, H, W):
            A = max(0, (H - original_array.shape[0]) // 2)
            AA = max(0, H - A - original_array.shape[0])
            B = max(0, (W - original_array.shape[1]) // 2)
            BB = max(0, W - B - original_array.shape[1])

            return np.pad(original_array, pad_width=((A, AA), (B, BB)), mode='constant')

        data, sampling_rate = librosa.load(wav_file, duration=SONG_DURATION, offset=0)
        data, index = librosa.effects.trim(data)  # Trim preceding and trailing zeros
        stft = padding(np.abs(librosa.stft(data, n_fft=255, hop_length=512)), FEATURE_HEIGHT, FEATURE_WIDTH)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='time', ax=ax)

        ax.set_title('Power Spectrogram (STFT): {}'.format(ensemble))
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def display_demos(self):
        """
        Method used to display example spectrograms and fourier transforms
        :return:
        :rtype:
        """

        piano_solo = "demo_data/1733.wav"
        string_quartet = "demo_data/1792.wav"
        accompanied_violin = "demo_data/2330.wav"

        self.display_stft(piano_solo, ensemble='Piano Solo')
        self.display_stft(string_quartet, ensemble='String Quartet')
        self.display_stft(accompanied_violin, ensemble='Accompanied Violin')

        self.display_mel_spec(piano_solo, ensemble='Piano Solo')
        self.display_mel_spec(string_quartet, ensemble='String Quartet')
        self.display_mel_spec(accompanied_violin, ensemble='Accompanied Violin')

    def run(self):
        """
        Primary method used to facilitate data processing, model creation, and plotting
        :return:
        :rtype:
        """

        # Filter dataset: include only desired ensemble types
        self.pre_process_metadata()

        for index, row in self.df.iterrows():
            id = str(row['id']) + ".wav"
            wav_file_path = os.path.join(WAV_FILE_PATH, id)

            images = self.get_features(wav_file_path)
            label = self.get_label(row['id'])
            label = self.label_codes[label]

            for image in images:
                # Split X and Y into test and train
                if random.random() > TEST_SPLIT:    # Test
                    self.X_test.append(image)
                    self.y_test.append(label)
                else:
                    self.X_train.append(image)
                    self.y_train.append(label)

        # Normalize
        self.normalize()

        # Generate model and compile. Fit model
        model = self.create_CNN()
        model.summary()
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train, epochs=NEPOCHS, validation_data=(self.X_test, self.y_test))

        loss, acc, val_loss, val_acc = history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']

        # Plot training and validation loss and accuracy
        self.plot(loss, acc, val_loss, val_acc)


def main():

    instrumentClassification = InstrumentClassification()
    instrumentClassification.display_demos()
    instrumentClassification.run()


if __name__ == "__main__":
    main()