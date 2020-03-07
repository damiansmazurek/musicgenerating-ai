import os
import logging
from train import train

#logging configuration
logging.basicConfig(format='%(asctime)s; %(levelname)s: %(message)s', level=logging.DEBUG)

#Get global env
TRAINING_SET_PATH= 'training_set' #os.getenv('TRAINING_SET_PATH')
MODEL_OUTPUT= 'model_output' #os.getenv('MODEL_OUTPUT')
N_FFT = os.getenv('N_FFT')
EPOCH = os.getenv('EPOCH')
BATCH_SIZE = os.getenv('BATCH_SIZE')
#for voice 512 for music 2048
N_FFT = 2048

train(TRAINING_SET_PATH, N_FFT, MODEL_OUTPUT,5,4,1)