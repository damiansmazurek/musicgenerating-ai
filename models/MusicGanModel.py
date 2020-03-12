import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Reshape, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from logging import log, info, debug 
import os
from utils import ModelsSufix

class GANMusicGenerator:
    def __init__(self, width, height, channels, ouputmodelpath):
        self.disc_output_model_path = ouputmodelpath + ModelsSufix.DICSR
        self.gen_output_model_path = ouputmodelpath + ModelsSufix.GEN
        self.width = width
        self.height = height
        self.channels = channels
        debug('Setting Adam optimizer')
        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        debug('Creating generator model.')
        self.g_model = self.__generator()
        debug('Compiling of model')
        self.g_model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        debug('Creating discriminator')
        self.d_optimizer = Adam(lr=0.004, beta_1=0.5, decay=8e-8)
        self.d_model = self.__discriminator()
        self.d_model.compile(loss='binary_crossentropy', optimizer=self.d_optimizer,metrics=['accuracy'])
        debug('Stacl model')
        self.stack_model = Sequential()
        self.stack_model.add(self.g_model)
        self.stack_model.add(self.d_model)
        self.stack_model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def __discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(self.width, self.height, self.channels)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        if os.path.exists(self.disc_output_model_path):
            info("Loading discriminator weights.")
            model.load_weights(self.disc_output_model_path)
        return model

    def __generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.height,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))
        if os.path.exists(self.gen_output_model_path):
            info("Loading generator weights.")
            model.load_weights(self.gen_output_model_path)
        return model

    def train(self, X_train, epochs=20000, batch = 32, save_interval = 100, discriminator_epoch_mul = 2, save_callback= None):
        for cnt in range(epochs):
            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            legit_data = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)
            
            # Generating preditions array with size of half batch.
            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), self.height))
            fake_data = self.g_model.predict(gen_noise)

            # Creating combined dataset with true data and fake ones
            x_combined_batch = np.concatenate((legit_data, fake_data))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            debug('Start training discriminator')
            d_loss = []
            for i in range(discriminator_epoch_mul):
                d_loss = self.d_model.train_on_batch(x_combined_batch, y_combined_batch)
                info('d_loss %f for true data and %f for fake data form discr_epoch_mul %d'%(d_loss[0], d_loss[1],i))
            debug('End training of discriminator for batch')

            # train generator
            noise = np.random.normal(0, 1, (batch, self.height))
            y_mislabled = np.ones((batch, 1))

            debug('Start training stacked model')
            g_loss = self.stack_model.train_on_batch(noise, y_mislabled)
            
            info('epoch: %d, [Discriminator :: d_loss: %f for true %f for fake], [ Generator :: loss: %f]' % (cnt, d_loss[0], d_loss[1], g_loss))

            if (cnt+1) % save_interval == 0:
                info('Saving model state after epoch:  %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
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
        