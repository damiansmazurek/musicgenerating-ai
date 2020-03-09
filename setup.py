import os
import logging
from train import train, generate



# APP AND HYPERPARAMETERS CONFIGURATION

# Main operation, you can set train or gen
CMD = os.getenv('CMD')

#for voice 512 for music 2048
N_FFT = int(os.getenv('N_FFT'))

# Number of sample for generating file (file lenght
SAMPLE_NUMBER = int(os.getenv('SAMPLE_NUMBER'))

# Path to directory, where model will be saved
MODEL_OUTPUT= os.getenv('MODEL_OUTPUT')

# Log level (if set to debug - debug log level on, esle info level log is set)
LOG_LEVEL = os.getenv('LOG_LEVEL')

# Logging configuration
level_logs = logging.INFO
if(LOG_LEVEL == 'debug'):
    level_logs = logging.DEBUG
logging.basicConfig(format='%(asctime)s; %(levelname)s: %(message)s', level=level_logs)

# Run specyfic functionality based on global command
if CMD == 'train':
    TRAINING_SET_PATH= os.getenv('TRAINING_SET_PATH')
    EPOCH = int(os.getenv('EPOCH'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    train(TRAINING_SET_PATH, N_FFT, MODEL_OUTPUT,EPOCH,BATCH_SIZE,1, SAMPLE_NUMBER)
else:
    OUTPUT_FILE = os.getenv('OUTPUT_FILE')
    generate(MODEL_OUTPUT,OUTPUT_FILE,N_FFT,SAMPLE_NUMBER)