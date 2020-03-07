import librosa
import numpy as np
from logging import log, info, debug, basicConfig, DEBUG, INFO

def wav2spect(filename, N_FFT):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, sr

def spect2wav(spectrum, sr, outfile):
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
    librosa.output.write_wav(outfile, x, sr)

def normalize_spectrums(spectrum_array,width, height):
    result_list = []
    for spect in spectrum_array:
        debug('Start to normalizing first spectrum of size: %s', spect.shape)
        reshaped_spect = np.resize(spect,(width,height))
        result_list.append(reshaped_spect)
        debug('Changed to array of size: %s', reshaped_spect.shape)
    return result_list