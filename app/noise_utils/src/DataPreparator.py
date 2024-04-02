import numpy as np
from perlin_noise import PerlinNoise
import librosa
import soundfile as sf
import os
import random
from tqdm import tqdm
from SignalGenerator import SignalGenerator

class DataPreparator:
    def __init__(self):
        self.generator = SignalGenerator()

    def generate_linear_dataset(self, size, length, noisemaker):
        """
        Сгенерировать датасет из слабонелинейных сигналов

        size: количество сигналов
        length: длина сигналов
        noisemaker: функция для создания шума
        return: чистые сигналы, искаженные сигналы
        """
        signals = np.array([])
        distored_signals = np.array([])
        for i in range(size):
            slope = random.uniform(-5, 5)
            intercept = random.uniform(-5, 5)
            signal = self.generator.generate_linear_signal(length, slope, intercept)
            delta = abs(max(signal) - min(signal))
            distored_signal, noise = noisemaker(signal, delta)
            signals = np.append(signals, signal)
            distored_signals = np.append(distored_signals, distored_signal)
        return distored_signals.reshape(-1, length), signals.reshape(-1, length)

    def generate_weakly_nonlinear_dataset(self, size, length, noisemaker):
        """
       Сгенерировать датасет из слабонелинейных сигналов

       size: количество сигналов
       length: длина сигналов
       noisemaker: функция для создания шума
       return: чистые сигналы, искаженные сигналы
       """
        signals = np.array([])
        distored_signals = np.array([])
        for i in range(size):
            phase = random.uniform(0, 5)
            signal = self.generator.generate_weakly_nonlinear_signal(length=length, phase=phase)
            # delta = (max(signal) - min(signal)) * 0.3
            distored_signal, noise = noisemaker(signal)
            signals = np.append(signals, signal)
            distored_signals = np.append(distored_signals, distored_signal)
        return distored_signals.reshape(-1, length), signals.reshape(-1, length), noise

    def generate_nonlinear_dataset(self, size, length, noisemaker):
        """
        Сгенерировать датасет из нелинейных сигналов

        size: количество сигналов
        length: длина сигналов
        noisemaker: функция для создания шума
        return: чистые сигналы, искаженные сигналы
        """
        signals = np.array([])
        distored_signals = np.array([])
        for i in range(size):
            frequency = 1
            amplitude = 5
            phase = random.uniform(0, 5)
            signal = self.generator.generate_nonlinear_signal(length, frequency, amplitude, phase)
            delta = (max(signal) - min(signal)) * 0.3
            distored_signal, noise = noisemaker(signal, delta)
            signals = np.append(signals, signal)
            distored_signals = np.append(distored_signals, distored_signal)
        return distored_signals.reshape(-1, length), signals.reshape(-1, length)

    def generate_noised_audio(self, input_file, output_file, noisemaker, noise_level=0.2):
        """
        Сгенерировать одно зашумленное аудио

        input_file: имя файла с чистым аудио
        output_file: имя файла для зашумленного аудио
        noisemaker: функция для генерации шума
        noise_level: интенсивность шума
        return: зашумленный файл (в виде массива чисел),
                частота дискретизации
        """
        audio, sr = librosa.load(input_file, sr=None)
        noise = np.random.normal(scale=noise_level, size=len(audio))
        noisy_audio, noise = noisemaker(audio, noise_level)
        max_value = np.max(np.abs(noisy_audio))
        if max_value > 1.0:
            noisy_audio /= max_value
        sf.write(output_file, noisy_audio, sr)
        return noisy_audio, sr

    def load_file(self, file):
        """
        Загрузить аудио

        file: имя файла
        return: аудио в виде массива чисел, частота дискретизации
        """
        audio, sr = librosa.load(file)
        return audio, sr

    def save_audio(self, audio, output_file, sr):
        """
        Сохранить аудио

        audio: аудио в виде массива чисел
        output_file: имя файла для записи
        sr: частота дискретизации
        """
        sf.write(output_file, audio, sr)

    def load_files_from_directory(self, directory):
        """
        Загрузить аудиофайлы из директории

        directory: директория с файлами
        return: массив из аудио, каждый из которых представлен в виде массива чисел,
                массив из названий файлов в директории,
                частота дискретизации аудиофайла
        """
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(str(directory), str(f)))]
        x = []
        sr = 0
        for file in files:
            audio, sr = self.load_file(os.path.join(str(directory), str(file)))
            x.append(audio)
        # display(x)
        # display(x.shape)
        return x, files, sr

    def generate_noised_audios(self, input_file, output_file, noisemaker, noise_level=0.2):
        """
        Сгенерировать зашумленные аудио

        input_file: директория с чистыми аудио
        output_file: директория для зашумленных аудио
        noisemaker: функция для генерации шума
        noise_level: интенсивность шума
        return: массив из зашумленных аудио, каждый из которых представлен в виде массива чисел,
                массив из чистых аудио, каждый из которых представлен в виде массива,
                частота дискретизации аудиофайла
        """
        data, files, sr = self.load_files_from_directory(input_file)
        if not os.path.exists(output_file):
            os.mkdir(output_file)
        noisy_audios = []
        for audio, file in tqdm(zip(data, files)):
            noisy_audio, noise = noisemaker(audio, noise_level)
            max_value = np.max(np.abs(noisy_audio))
            if max_value > 1.0:
                noisy_audio /= max_value
            noisy_audios.append([noisy_audio])
            sf.write(os.path.join(str(output_file), str(file)), noisy_audio, sr)
        noisy_audios = noisy_audios
        return noisy_audios, data, sr

    def reshape_data(self, data, window_size=1000):
        """
        Делит поступающий ввод на части фиксированной длины

        data: двумерный массив
        window_size: длина частей нового массива
        return: двумерный массив, где каждый массив длины window_size
        """
        res = []
        for item in data:
            for i in range(0, len(item) - window_size, window_size):
                chunk = item[i:i + window_size]
                res.append(chunk)
        return np.array(res)