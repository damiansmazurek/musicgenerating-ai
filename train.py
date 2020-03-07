from utils import wav2spect, spect2wav, normalize_spectrums
from models.MusicGanModel import CNNDiscriminatorGAN
import numpy as np
import os
from logging import log, info, debug, basicConfig, DEBUG, INFO


def train(training_set_path, N_FFT, model_path, epochs=1000, batch=4, save_interval=100):
    #Open files and fft them
    info('Training started')
    train_data = []
    sr_lists = []
    max_height = 0
    for filename in os.listdir(training_set_path):
        if "wav" not in filename: 
            continue
        info('Creating FFT for file: %s',filename)
        spectrum, sr = wav2spect(training_set_path+'/'+filename, N_FFT)
        if len(spectrum[0]) > max_height:
            debug("Change max_height to %d", len(spectrum[0]))
            max_height = len(spectrum[0])
        train_data.append(np.asarray(spectrum))
        sr_lists.append(sr)

    #Set parameters for model
    width = len(train_data[0])
    height = 1025#max_height
    channels = 1
    info("Setting data size to: %d x %d x %d", width,height,channels)
    
    train_data = normalize_spectrums(train_data,width,height)

    train_data_norm= np.asarray(train_data)
    debug("data size set to %s",train_data_norm.shape)

    debug("Generating GAN")
    gan = CNNDiscriminatorGAN(width,height,channels,model_path)

    #train model
    info("Start training")
    gan.train(train_data_norm,epochs, batch, save_interval)

    #generate musicspect2wav(gan.generate(),spectrum[1],"data-gen.wav")

