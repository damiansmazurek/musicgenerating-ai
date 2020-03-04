import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

class CNNDiscriminatorGAN:
    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels
        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        self.g_model = self.__generator()
        self.g_model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.d_model = self.__discriminator()
        self.d_model.compile(loss='binary_crossentropy', optimizer=self.optimizer,metrics=['accuracy'])
        self.stack_model = Sequential()
        self.stack_model.add(self.g_model)
        self.stack_model.add(self.d_model)
        self.stack_model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def __discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.build()
        model.summary()
        return model

    
    def __generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
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
        return model

    def train(self, X_train, epochs=20000, batch = 32, save_interval = 100):

        for cnt in range(epochs):

            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            legit_data = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
            syntetic_data = self.g_model.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_data, syntetic_data))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            d_loss = self.d_model.train_on_batch(x_combined_batch, y_combined_batch)


            # train generator

            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stack_model.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

           # if cnt % save_interval == 0:
            #    self.plot_images(save2file=True, step=cnt)

    def generate(self):
         gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
         return self.g_model.predict(gen_noise)
   