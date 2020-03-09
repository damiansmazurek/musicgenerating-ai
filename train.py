from utils import wav2spect, spect2wav, normalize_spectrums
from models.MusicGanModel import CNNDiscriminatorGAN
import numpy as np
import os
from logging import log, info, debug, basicConfig, DEBUG, INFO
import tensorflow as tf

def train(training_set_path, N_FFT, model_path, epochs=1000, batch=4, save_interval=100, max_height = 1025):
    #Open files and fft them
    info('Training started')
    train_data = []
    sr_lists = []
    for filename in os.listdir(training_set_path):
        if "wav" not in filename: 
            continue
        info('Creating FFT for file: %s',filename)
        spectrum, sr = wav2spect(training_set_path+'/'+filename, N_FFT)
        if len(spectrum[0]) > max_height:
            debug("Change max_height to %d", len(spectrum[0]))
        train_data.append(np.asarray(spectrum))
        debug("SR", sr)
        sr_lists.append(sr)

    #Set parameters for model
    width = len(train_data[0])
    height = max_height
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

def generate(model_path, otputfile, N_FFT, sample_number = 1025, sr = 22050):
    batch = 4
    gan = tf.keras.models.load_model(model_path+"/gen.h5")
    spectrum = gan.predict(np.random.normal(0, 1, (np.int64(batch/2), sample_number)))
    i = 0
    for single in spectrum:
        debug(single)
        i = i+1
        spect2wav(single.reshape((single.shape[0],single.shape[1])),22050,otputfile+str(i)+'.wav', N_FFT)