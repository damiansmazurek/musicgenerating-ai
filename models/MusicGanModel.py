import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Conv2DTranspose, MaxPool2D, Dropout, BatchNormalization, Reshape, ZeroPadding2D, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from logging import log, info, debug 
import os
from utils import ModelsSufix, plot_spectrum
import math

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05, mean=0, seed=42) 
class GANMusicGenerator:
    def __init__(self, width, height, channels, ouputmodelpath):
        self.disc_output_model_path = ouputmodelpath + ModelsSufix.DICSR
        self.gen_output_model_path = ouputmodelpath + ModelsSufix.GEN
        self.width = width
        self.height = height
        self.channels = channels
        self.optimizer = Adam(lr=1e-4, beta_1=0.5, decay=8e-8)
        self.g_model = self.__generator()
        self.g_model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        self.d_optimizer = Adam(lr=1e-4)
        self.d_model = self.__discriminator()
        self.d_model.compile(optimizer=self.d_optimizer, loss = 'binary_crossentropy')
        self.stackmodel = Sequential()
        self.stackmodel.add(self.g_model)
        self.d_model.trainable=False
        self.stackmodel.add(self.d_model)
        self.stackmodel.compile(optimizer=self.optimizer, loss= 'binary_crossentropy')

    def __discriminator(self):
        model = Sequential([
            GaussianNoise(0.3,input_shape=(self.width, self.height, self.channels)),
            Conv2D(filters=64, kernel_size=(5, 5), padding='same', kernel_initializer = weight_initializer),
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(filters=128, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_initializer = weight_initializer),
            LeakyReLU(),
            Dropout(0.3),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        info(model.summary())
        if os.path.exists(self.disc_output_model_path):
            info("Loading discriminator weights.")
            model.load_weights(self.disc_output_model_path)
        return model

    def __generator(self):
        model = Sequential([
            Dense(math.ceil(self.width/8)*math.ceil(self.height/8)*256, input_shape = [100]),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((math.ceil(self.width/8), math.ceil(self.height/8), 256)),
            Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, output_padding=(self.height%2,self.width%2)),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, output_padding=(self.height%2,self.width%2)),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh', output_padding=(self.height%2,self.width%2))
        ])
        info('Model required shape shape: %s'%(str((None, self.width, self.height, self.channels))))
        info('Model output shape: %s'%(str(model.output_shape)))
        #assert model.output_shape == (None, self.width, self.height, 1)
        info(model.summary())
        if os.path.exists(self.gen_output_model_path):
            info("Loading generator weights.")
            model.load_weights(self.gen_output_model_path)
        return model

    def train(self, X_train, epochs=20000, batch = 1, save_interval = 100, smoothing_factor = 0.1, save_callback= None):
        for cnt in range(epochs):
            # for single file
            random_index = 0
            
            # for many files in dataset
            if len(X_train) != 1:
                random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            
            # Prepare training set.
            legit_data = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels) 
            
            # Generating preditions array with size of half batch.
            gen_noise = np.random.normal(0, 1, (np.int64(batch/2),100))
            gen_data = self.g_model.predict(gen_noise)
            acc_real = self.d_model.train_on_batch(legit_data,self.__label_smoothing(np.ones(np.int64(batch/2)),0.3))
            acc_fake = self.d_model.train_on_batch( gen_data, self.__label_smoothing(np.zeros(np.int64(batch/2)),0.3))
            gen_training_noise = np.random.normal(0, 1, (batch,100))
            gen_loss = self.stackmodel.train_on_batch(gen_training_noise, np.ones(np.int64(batch)))
            
            info('epoch: %d from %d, [Discriminator :: d_loss_real: %f, d_loss_fake: %f], [ Generator :: loss: %f] ' % (cnt,epochs,acc_real, acc_fake, gen_loss))
            if(cnt == 0):
                self.__generate_and_save_image(cnt,gen_training_noise)
                
            if (cnt+1) % save_interval == 0:
                self.__generate_and_save_image(cnt,gen_training_noise)
                info('Saving model state after epoch:  %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, 0, gen_loss))
                self.__run_save_model(save_callback)
                
        info('Training completed, exporting models')
        self.__run_save_model(save_callback)

    def __run_save_model(self, callback):
        self.d_model.save(self.disc_output_model_path)
        self.g_model.save(self.gen_output_model_path)
        info('Models saved locally....')
        if callback != None:
            info('Start uploading models to cloud...')
            callback(self.gen_output_model_path, self.disc_output_model_path)
    
    def __generate_and_save_image(self, epoch, noise):
        generated_data = self.g_model.predict(noise)
        fake_output = self.d_model.predict(generated_data)
        plot_spectrum(np.squeeze(generated_data[0]),'tmp/epoch_'+str(epoch)+'.png')



    def __label_smoothing(self, labels, smoothing_factor):
        # smooth the labels
        labels *= (1 - random.uniform(0, smoothing_factor))
        # returned the smoothed labels
        return labels