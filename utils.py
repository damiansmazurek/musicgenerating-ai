import librosa
import numpy as np
from logging import log, info, debug, basicConfig, DEBUG, INFO
from google.cloud import storage
from google.api_core.exceptions import NotFound

def wav2spect(filename, N_FFT):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, sr

def spect2wav(spectrum, sr, outfile, N_FFT):
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

def download_blobs(source_bucket_name, local_directory):
    storage_client = storage.Client()
    blob_list = storage_client.list_blobs(source_bucket_name)
    i = 0
    for blob_item in blob_list: 
        with open(local_directory+'/sample'+str(i)+'.wav','wb') as file_obj:
            storage_client.download_blob_to_file(blob_item, file_obj)
        i = i+1

def download_model(model_bucket_name, model_name, model_local_path):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        blob_disc = bucket.blob(model_bucket_name+ModelsSufix.DICSR)
        blob_gen = bucket.blob(model_bucket_name+ModelsSufix.GEN)
        blob_disc.download_to_filename(model_local_path+ModelsSufix.DICSR)
        blob_gen.download_to_filename(model_local_path+ModelsSufix.GEN)
    except NotFound:
        info('No model repository found.')

def upload_model(model_bucket, model_blob_path, model_path_disc, model_path_gen):
    storage_client = storage.Client()
    bucket = storage_client.bucket(model_bucket)
    blob_disc = bucket.blob(model_blob_path + ModelsSufix.DICSR)
    blob_gen = bucket.blob(model_blob_path + ModelsSufix.GEN)
    blob_gen.upload_from_filename(model_path_gen)
    blob_disc.upload_from_filename(model_path_disc)

def upload_blob_to_bucket(filepath, outpu_bucket_name ):
    storage_client = storage.Client()
    bucket = storage_client.bucket(outpu_bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

class ModelsSufix:
    GEN = '/gen.h5'
    DICSR = '/desc.h5'