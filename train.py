from utils import wav2spect, spect2wav
from models.MusicGanModel import CNNDiscriminatorGAN

#for voice 512 for music 2048
N_FFT = 2048
spectrum = wav2spect('data-sample.wav', N_FFT)
width = len(spectrum[0])
height = len(spectrum[0][0])
channels = 1
print(len(spectrum[0][0]))

gan = CNNDiscriminatorGAN(width,height,channels)

#train model
gan.train(spectrum[0])

#generate musicspect2wav(gan.generate(),spectrum[1],"data-gen.wav")