import numpy as np
from perlin_noise import PerlinNoise
import librosa
import soundfile as sf
import os
import random
from tqdm import tqdm


class Noisemaker:
    def add_random_noise(self, signal, noise_level=0.002):
        noise = np.random.normal(scale=noise_level, size=len(signal))
        noisy_signal = signal + noise
        max_value = np.max(np.abs(noisy_signal))
        if max_value > 1.0:
            noisy_signal /= max_value
        return noisy_signal, noise

    def add_perlin_noise(self, signal, noise_level=3):
        noise_f = PerlinNoise()
        noise = np.array([noise_f(i * signal[i]) * noise_level for i in range(len(signal))])
        noisy_signal = signal + noise
        # max_value = np.max(np.abs(noisy_signal))
        # if max_value > 1.0:
        #     noisy_signal /= max_value
        return noisy_signal, noise


class SignalGenerator:
    def generate_linear_signal(self, length, slope=1, intercept=0):
        x = np.arange(length)
        signal = slope * x + intercept
        return signal

    def generate_weakly_nonlinear_signal(self, length, frequency=1, amplitude=5, phase=0, nonlinearity=0.1):
        t = np.linspace(0, 2 * np.pi * frequency, length)
        signal = amplitude * (np.sin(t + phase) + nonlinearity * np.sin(2 * t + 2 * phase))
        return signal

    def generate_nonlinear_signal(self, length, frequency=1, amplitude=5, phase=0):
        t = np.linspace(0, 2 * np.pi * frequency, length)
        signal = amplitude * np.sin(t + phase) ** 2
        return signal

    def generate_linear_signals(self, length, slope=1, intercept=0):
        x = np.arange(length)
        signal = slope * x + intercept
        return signal

    def generate_weakly_nonlinear_signals(self, length, frequency=1, amplitude=1, phase=0, nonlinearity=0.1):
        t = np.linspace(0, 2 * np.pi * frequency, length)
        signal = amplitude * (np.sin(t + phase) + nonlinearity * np.sin(2 * t + 2 * phase))
        return signal


class DataPreparator:
    def __init__(self):
        self.generator = SignalGenerator()

    def generate_linear_dataset(self, size, length, noisemaker):
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
        audio, sr = librosa.load(input_file, sr=None)
        noise = np.random.normal(scale=noise_level, size=len(audio))
        noisy_audio, noise = noisemaker(audio, noise_level)
        max_value = np.max(np.abs(noisy_audio))
        if max_value > 1.0:
            noisy_audio /= max_value
        sf.write(output_file, noisy_audio, sr)
        return noisy_audio, sr

    def load_file(self, file):
        audio, sr = librosa.load(file)
        return audio, sr

    def save_audio(self, audio, output_file, sr):
        sf.write(output_file, audio, sr)

    def load_files_from_directory(self, directory):
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
        res = []
        for item in data:
            for i in range(0, len(item) - window_size, window_size):
                chunk = item[i:i + window_size]
                res.append(chunk)
        return np.array(res)
