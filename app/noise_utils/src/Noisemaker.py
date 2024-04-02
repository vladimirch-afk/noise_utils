import numpy as np
from perlin_noise import PerlinNoise

class Noisemaker:
    def add_random_noise(self, signal, noise_level=0.002):
        """
        Добавить случайный шум

        signal: сигнал (в том числе аудиосигнал) в виде массива чисел
        noise_level: интенсивность шума
        return: зашумленный сигнал, прибавленный шум
        """
        noise = np.random.normal(scale=noise_level, size=len(signal))
        noisy_signal = signal + noise
        max_value = np.max(np.abs(noisy_signal))
        if max_value > 1.0:
            noisy_signal /= max_value
        return noisy_signal, noise

    def add_perlin_noise(self, signal, noise_level=3):
        """
        Добавить шум Перлина

        signal: сигнал (в том числе аудиосигнал) в виде массива чисел
        noise_level: интенсивность шума
        return: зашумленный сигнал, прибавленный шум
        """
        noise_f = PerlinNoise()
        noise = np.array([noise_f(i * signal[i]) * noise_level for i in range(len(signal))])
        noisy_signal = signal + noise
        # max_value = np.max(np.abs(noisy_signal))
        # if max_value > 1.0:
        #     noisy_signal /= max_value
        return noisy_signal, noise

