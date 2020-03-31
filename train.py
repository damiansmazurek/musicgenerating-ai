from utils import wav2spect, spect2wav, normalize_spectrums, upload_model, ModelsSufix, plot_spectrum
from models.MusicGenerator import MusicGenerator
from models.MusicRNNModel import RNNMusicGenerator
import numpy as np
import os
from logging import log, info, debug, basicConfig, DEBUG, INFO
import tensorflow as tf

class TrainingController:
    def __init__(self, model_bucket= None, model_blob_path= None):
        self.model_bucket = model_bucket
        self. model_blob_path = model_blob_path

    def save_callback(self, model_path_gen, model_path_disc):
        info('Uploading model to bucket %s'%(self.model_bucket))
        upload_model(self.model_bucket, self.model_blob_path, model_path_disc, model_path_gen)

    def train(self, training_set_path, N_FFT, model_path, epochs=1000, batch=4, save_interval=100, max_height = 1025, smoothing_factor = 0.1):
        #Open files and fft them
        info('Training started')
        train_data = []
        sr_lists = []
        for filename in os.listdir(training_set_path):
            if "wav" not in filename: 
                continue
            info('Creating FFT for file: %s',filename)
            spectrum, sr = wav2spect(training_set_path+'/'+filename, N_FFT)
            plot_spectrum(spectrum,filename+'.png')
            if len(spectrum[0]) > max_height:
                debug("Change max_height to %d", len(spectrum[0]))
            train_data.append(np.asarray(spectrum))
            debug("SR", sr)
            sr_lists.append(sr)

        # Set parameters for model
        width = len(train_data[0])
        height = max_height
        channels = 1
        
        info("Setting data size to: %d x %d x %d", width,height,channels)
        train_data_norm = normalize_spectrums(train_data,width,height)
        info("Train data size: %s"%(str(train_data_norm.shape)))
        
        # Check if batch is not bigger then training set.
        if batch > len(train_data_norm):
            batch = len(train_data_norm)
            info('Batch is bigger then dataset sample, changing batch size to %d'%(batch))

        info("Generating GAN model")
        gan = RNNMusicGenerator(width,height,channels,model_path)

        #train model
        info("Start training")
        save_bucket_callback = None
        if self.model_bucket != None:
            info('Exporting models mode to GCP bucket is turned on')
            save_bucket_callback = self.save_callback
        gan.train(train_data_norm,epochs, batch, save_interval, smoothing_factor, save_bucket_callback)

    def generate(self, model_path, otputfile, N_FFT, path_to_start_sample, sample_number = 1025, sr = 22050, lenght= 10):
        generator = MusicGenerator(model_path)
        first_part, sr = wav2spect(path_to_start_sample, N_FFT)
        width = len(first_part)
        fft_data_norm = normalize_spectrums([first_part],width,sample_number)
        spectrum = generator.generate_music(width,sample_number,fft_data_norm[0].reshape(1,width,sample_number,1),lenght)
        spect2wav(spectrum, sr, otputfile+'/gen_music.wav', N_FFT)


