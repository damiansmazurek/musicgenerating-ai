FROM tensorflow/tensorflow:2.0.1-gpu-py3
COPY . . 
RUN pip install -r requirements.txt
ENV CMD = 'train'
ENV N_FFT = '2048'
ENV SAMPLE_NUMBER = '1025'
ENV MODEL_OUTPUT = 'MODEL_OUTPUT'
ENV LOG_LEVEL = 'info'
ENV TRAINING_SET_PATH = 'training_set'
ENV EPOCH = '50000'
ENV BATCH_SIZE = '4'
ENV SAVE_INTERVAL = '100'
ENV MODEL_BUCKET = 'none'
ENV MODEL_NAME = 'musicgenai'
ENV TRAINING_SET_BUCKET = 'none'
ENV OUTPUT_FILE = 'output/generated'
ENV DISCR_EPOCH_MUL = '10'
CMD python setup.py