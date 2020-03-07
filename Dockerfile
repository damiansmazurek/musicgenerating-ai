FROM tensorflow/tensorflow
COPY . .
RUN pip install -r requirements.txt
