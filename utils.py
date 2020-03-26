import librosa
import numpy as np
from logging import log, info, debug, basicConfig, DEBUG, INFO
from google.cloud import storage
from google.api_core.exceptions import NotFound
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_spectrum(gen_spectrum, image_name):
    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("CNN Voice Transfer Result")
    plt.imsave(image_name, gen_spectrum[:400, :])

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
    result_list = None
    for spect in spectrum_array:
        spect_t = np.transpose(spect)
        if result_list == None:
            result_list = split_single_fft(spect_t, height)
        else:
            result_list = np.concatenate(result_list,split_single_fft(spect, height))
    info('Size of the array: %s'%(str(result_list.shape)))
    return result_list

def split_single_fft(single_item, sample_number):
    i = 0
    result_samples = []
    info('Spliting data of size %s'%(str(len(single_item))))
    while (i+1)*sample_number < len(single_item):
        result_samples.append(np.transpose(single_item[i*sample_number:(i+1)*sample_number]))
        i = i + 1
        info('Result sample size %s'%(str(result_samples)))
    return np.array(result_samples)

   

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
        info('Downloading models from GCP storage repository')
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        blob_disc = bucket.blob(model_name+ModelsSufix.DICSR)
        if blob_disc.exists():
            blob_disc.download_to_filename(model_local_path+ModelsSufix.DICSR)
        else:
            info('No discriminator model found in cloud storage')
        blob_gen = bucket.blob(model_name+ModelsSufix.GEN)
        if blob_gen.exists():
            blob_gen.download_to_filename(model_local_path+ModelsSufix.GEN)
        else:
            info('No generator model found in cloud storage')  
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