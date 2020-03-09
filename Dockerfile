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
CMD python setup.py