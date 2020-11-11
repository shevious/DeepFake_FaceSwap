#!/usr/bin/env python
# coding: utf-8

# # Deepfakes 101 - Model Training
# 
# This notebook aims to train our autoencoder on the clean training data that have been previously generated.
# 
# Deepfake generated  is  done through an  autoencoder. 
# 
# The face of the actor 1 is cropped and aligned to his face. Then the autoencoder learn how to encode and decode (reconstruct) the face. The goal here is to minimize the reconstruction error, defined as the mean absolute error (pixel to pixel). This is then connected to the decoder of actor B, generating a composite with the characterstics of B while still maintaining the likeness of A.
# 
# More details can be found on the original full tutorial.
# 
# 
# 

# We load the various dependencies

# In[ ]:
import os


os.system('mkdir -p saved_model')

#Download auxilliary components first


#os.system('gdown https://drive.google.com/uc?id=1O0jrWmtAoSN-W8AwmO0GrqhcYPzkZB2Y')
  
#os.system('unzip deepfake_aux.zip')

#Download extracted datasets from pt.1
#Tom

#os.system('gdown https://drive.google.com/uc?id=1U7WkAUzGFuSIil2NvaZd3dpCzn34BwlQ')
#Nic

#os.system('gdown https://drive.google.com/uc?id=1txJSjz3GYjDm9OgMI-YTW8yWvrzVJED3')
#ChrisOLD
#os.system('gdown https://drive.google.com/uc?id=1Dpm6QssNM7K3vl_-9n9YrIBqLWFi1OFx')
  
#os.system('unzip extracted_tom.zip')
#os.system('unzip extracted_nic.zip')
#os.system('unzip extracted_chris.zip')
  


# In[ ]:


#Install dependencies if missing

#os.system('pip install face_recognition')

#os.system('pip install scandir')
#os.system('pip install h5py')
#os.system('pip install opencv-python')
#os.system('pip install scikit-image')
#os.system('pip install dlib')
#os.system('pip install tqdm')

#Past 2.2.0, Keras's normalize_data_format method was moved to each specific layer lib.
#For convenience, we'll just use 2.2.0
#os.system('pip install --force-reinstall keras==2.2.0')


# In[ ]:


#Download pretrained weight to speed convergence. Weights2 from FakeAPP

#!gdown https://drive.google.com/uc?id=1ir6KEzNp63ZzWwIDjKcO7hgpuKOE7sil 

#os.system('gdown https://drive.google.com/uc?id=1biKvZSxV-Dx9mUX-UMaFcZ6J8OMQ9MHQ')
#os.system('unzip weights2.zip')


# In[ ]:


import cv2
import numpy
import time

from pathlib import Path
from scandir import scandir

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from lib_1.PixelShuffler import PixelShuffler
import time
import numpy
from lib_1.training_data import minibatchAB, stack_images



# We define the caracteristics of our network:
#    * sav_Model: the location of the model (it needs to be the absolute path)
#    * pretrained_weight: transfer learning can be used to speed up the learning process
#    * sav_Model: the location of the model (it needs to be the absolute path)
#    * image_actor_A_directory: the location of the training data for the actor A
#    * image_actor_B_directory: the location of the training data for the actor B
#    * batch_size: the batch size (1<< batch_size <<your training dataset size)
#        * a small batch_size enables poor hardware to run the training but the gradiant might be a bit noisy. I advice you to pick a batch size of 32 or 64
#    * save_interval: it define at which interval of time Keras should save your model
#    * IMAGE_SHAPE: the shape of the input image
#        * it must be consistant with the one you use during the prediction
#    * ENCODER_DIM: the dimension of the encoding in the autoencoder
#        * it must be consistant with the one you use during the prediction
#    
# I invite you to try different parameter until you get a satifying result.

# In[ ]:


#Define all directories

sav_Model="../content/saved_model/"
#pretrained_weight="../content/weight"
pretrained_weight="../content/saved_model/"
#image_actor_A_directory="../content/chris" #ORIGINAL
#image_actor_B_directory="../content/nic" #TARGET TO REPLACE WITH
image_actor_A_directory="../content/obama-cut" #ORIGINAL
image_actor_B_directory="../content/clinton-cut" #TARGET TO REPLACE WITH
batch_size=32
save_interval=500
ENCODER_DIM = 1024


#DON'T MODIFY
image_extensions = [".jpg", ".jpeg", ".png"]
encoderH5 = '/encoder.h5'
decoder_AH5 = '/decoder_A.h5'
decoder_BH5 = '/decoder_B.h5'
IMAGE_SHAPE = (64, 64, 3)


# We define the model and its associated trainer.

# In[ ]:


class dfModel():
    def __init__(self):

        self.model_dir = sav_Model
        self.pretrained_weight=pretrained_weight
        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    
    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999) #orig adam 5e-5
        x = Input(shape=IMAGE_SHAPE)

        self.autoencoder_A = KerasModel(x, self.decoder_A(self.encoder(x)))
        self.autoencoder_B = KerasModel(x, self.decoder_B(self.encoder(x)))
        print(self.encoder.summary())
        print(self.decoder_A.summary())


        self.autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

    def converter(self, swap):
        autoencoder = self.autoencoder_B if not swap else self.autoencoder_A 
        return lambda img: autoencoder.predict(img)

    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    def upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            #Pixelshufflers job here is analoguous to upsampling2d
            return x
        return block
#Note how no maxpooling after every layer here, we are generating a WARPED image, not a lower dimensional representation first.
    def Encoder(self):
        input_ = Input(shape=IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(ENCODER_DIM)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        #Passed flattened X input into 2 dense layers, 1024 and 1024*4*4
        x = Reshape((4, 4, 1024))(x)
        #Reshapes X into 4,4,1024
        x = self.upscale(512)(x)
        return KerasModel(input_, x)

    def Decoder(self):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale(256)(x) #Actually 1024 given filters*4
        x = self.upscale(128)(x) #Actually 512
        x = self.upscale(64)(x) #Actually 256
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return KerasModel(input_, x)
        
    def load(self, swapped):
        (face_A,face_B) = (decoder_AH5, decoder_BH5) if not swapped else (decoder_BH5, decoder_AH5)

        try:
            self.encoder.load_weights(self.pretrained_weight + encoderH5)
            self.decoder_A.load_weights(self.pretrained_weight + face_A)
            self.decoder_B.load_weights(self.pretrained_weight + face_B)
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(self.model_dir + encoderH5)
        self.decoder_A.save_weights(self.model_dir + decoder_AH5)
        self.decoder_B.save_weights(self.model_dir + decoder_BH5)
        print('saved model weights')

class Trainer():
    def __init__(self, model, fn_A, fn_B, batch_size=64):
        self.batch_size = batch_size
        self.model = model
        self.images_A = minibatchAB(fn_A, self.batch_size)
        self.images_B = minibatchAB(fn_B, self.batch_size)

    def train_one_step(self, iter):
        epoch, warped_A, target_A = next(self.images_A)
        epoch, warped_B, target_B = next(self.images_B)

        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
        print("[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"), iter, loss_A, loss_B),
            end='\r')


# In[ ]:


def get_image_paths(directory):
    return [x.path for x in scandir(directory) if
     any(map(lambda ext: x.name.lower().endswith(ext), image_extensions))]


# We launch the training. We will try to minimize th loss of the auto encoder A and B.

# In[ ]:



model = dfModel()
model.load(swapped=False)
model.summary()


# In[ ]:


print('Loading data, this may take a while...')
# this is so that you can enter case insensitive values for trainer
#1407 STOPPED AT 63 EPOCHS Disabled load to do it ourselves

model = dfModel()
model.load(swapped=False)



images_A = get_image_paths(image_actor_A_directory)
images_B = get_image_paths(image_actor_B_directory)
trainer = Trainer(model,images_A,images_B,batch_size=batch_size)

for epoch in range(0, 1000000):

    save_iteration = epoch % save_interval == 0

    trainer.train_one_step(epoch)

    if save_iteration:
        model.save_weights()
        
    


# In[ ]:





# In[ ]:




