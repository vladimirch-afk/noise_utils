import numpy as np

class SignalGenerator:
    def generate_linear_signal(self, length, slope=1, intercept=0):
        """
        Сгенерировать линейный сигнал y = Ax + b

        length: длина сигнала (количество значений)
        slope: коэффициент A
        intercept: сдвиг по оси Оу, кофф. b
        return: массив со значениями сигнала
        """
        x = np.arange(length)
        signal = slope * x + intercept
        return signal

    def generate_weakly_nonlinear_signal(self, length, frequency=1, amplitude=5, phase=0, nonlinearity=0.1):
        """
        Сгенерировать слабонелинейный сигнал

        length: длина сигнала (количество значений)
        frequency: частота сигнала
        amplitude: амплитуа сигнала
        phase: фаза сигнала
        nonlinearity: коэфф. для нелинейности сигнала
        return: массив со значениями сигнала
        """
        t = np.linspace(0, 2 * np.pi * frequency, length)
        signal = amplitude * (np.sin(t + phase) + nonlinearity * np.sin(2 * t + 2 * phase))
        return signal

    def generate_nonlinear_signal(self, length, frequency=1, amplitude=5, phase=0):
        """
       Сгенерировать слабонелинейный сигнал

       length: длина сигнала (количество значений)
       frequency: частота сигнала
       amplitude: амплитуа сигнала
       phase: фаза сигнала
       return: массив со значениями сигнала
       """
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