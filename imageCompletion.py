#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import imghdr
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.engine.network import Network
from keras.utils import generic_utils
from keras.layers import Reshape, Lambda, Conv2D, Conv2DTranspose, Flatten, Activation, Dense, Input
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as mplt


# In[7]:


def Generative(input_shape=(256, 256, 3)):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=1, padding='same',dilation_rate=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(8, 8)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(16, 16)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, strides=1, padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model


# In[ ]:


def Discriminative(global_shape=(256, 256, 3), local_shape=(128, 128, 3)):
    def imgCrop(img, crop):
        return tf.image.crop_to_bounding_box(img, crop[1], crop[0], crop[3] - crop[1], crop[2] - crop[0])

    inputLayer = Input(shape=(4,), dtype='int32')
    cropping = Lambda(lambda x: K.map_fn(lambda y: imgCrop(y[0], y[1]), elems=x, dtype=tf.float32), output_shape=local_shape)
    g_img = Input(shape=global_shape)
    l_img = cropping([g_img, inputLayer])

    DisNetwork = Conv2D(64, kernel_size=5, strides=2, padding='same')(l_img)
    DisNetwork = BatchNormalization()(DisNetwork)
    DisNetwork = Activation('relu')(DisNetwork)
    DisNetwork = Conv2D(128, kernel_size=5, strides=2, padding='same')(DisNetwork)
    DisNetwork = BatchNormalization()(DisNetwork)
    DisNetwork = Activation('relu')(DisNetwork)
    DisNetwork = Conv2D(256, kernel_size=5, strides=2, padding='same')(DisNetwork)
    DisNetwork = BatchNormalization()(DisNetwork)
    DisNetwork = Activation('relu')(DisNetwork)
    DisNetwork = Conv2D(512, kernel_size=5, strides=2, padding='same')(DisNetwork)
    DisNetwork = BatchNormalization()(DisNetwork)
    DisNetwork = Activation('relu')(DisNetwork)
    DisNetwork = Conv2D(512, kernel_size=5, strides=2, padding='same')(DisNetwork)
    DisNetwork = BatchNormalization()(DisNetwork)
    DisNetwork = Activation('relu')(DisNetwork)
    DisNetwork = Flatten()(DisNetwork)
    DisNetwork = Dense(1024, activation='relu')(DisNetwork)

    GenNetwork = Conv2D(64, kernel_size=5, strides=2, padding='same')(g_img)
    GenNetwork = BatchNormalization()(GenNetwork)
    GenNetwork = Activation('relu')(GenNetwork)
    GenNetwork = Conv2D(128, kernel_size=5, strides=2, padding='same')(GenNetwork)
    GenNetwork = BatchNormalization()(GenNetwork)
    GenNetwork = Activation('relu')(GenNetwork)
    GenNetwork = Conv2D(256, kernel_size=5, strides=2, padding='same')(GenNetwork)
    GenNetwork = BatchNormalization()(GenNetwork)
    GenNetwork = Activation('relu')(GenNetwork)
    GenNetwork = Conv2D(512, kernel_size=5, strides=2, padding='same')(GenNetwork)
    GenNetwork = BatchNormalization()(GenNetwork)
    GenNetwork = Activation('relu')(GenNetwork)
    GenNetwork = Conv2D(512, kernel_size=5, strides=2, padding='same')(GenNetwork)
    GenNetwork = BatchNormalization()(GenNetwork)
    GenNetwork = Activation('relu')(GenNetwork)
    GenNetwork = Conv2D(512, kernel_size=5, strides=2, padding='same')(GenNetwork)
    GenNetwork = BatchNormalization()(GenNetwork)
    GenNetwork = Activation('relu')(GenNetwork)
    GenNetwork = Flatten()(GenNetwork)
    GenNetwork = Dense(1024, activation='relu')(GenNetwork)

    x = concatenate([DisNetwork, GenNetwork])
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, inputLayer], outputs=x)


# In[8]:


if __name__ == "__main__":
    generator = Generative()
    discriminator = Discriminative()

class Data(object):
    def __init__(self, root_dir, imgSize, locSize):
        self.locSize = locSize
        self.imgSize = imgSize
        self.reset()
        self.fileList = []
        for root, dirs, files in os.walk(root_dir):
            for File in files:
                imgPath = os.path.join(root, File)
                if imghdr.what(imgPath) is None:
                    continue
                self.fileList.append(imgPath)
    
    def reset(self):
        self.images = []
        self.points = []
        self.masks = []


    def __len__(self):
        return len(self.fileList)
    
    def flow(self, batchSize, minBlock=64, maxBlock=128):
        np.random.shuffle(self.fileList)
        for f in self.fileList:
            i = cv2.imread(f)
            i = cv2.resize(i, self.imgSize)[:, :, ::-1]
            self.images.append(i)
            
            w, h = np.random.randint(minBlock, maxBlock, 2)
            a1 = np.random.randint(0, self.imgSize[0] - self.locSize[0] + 1)
            b1 = np.random.randint(0, self.imgSize[1] - self.locSize[1] + 1)
            
            a2,b2 = np.array([a1, b1]) + np.array(self.locSize)
            self.points.append([a1, b1, a2, b2])

            
            p1 = a1 + np.random.randint(0, self.locSize[0] - w)
            q1 = b1 + np.random.randint(0, self.locSize[1] - h)
            p2 = p1 + w
            q2 = q1 + h

            m = np.zeros((self.imgSize[0], self.imgSize[1], 1), dtype=np.uint8)
            m[q1:q2 + 1, p1:p2 + 1] = 1
            self.masks.append(m)

            if len(self.images) == batchSize:
                inputs = np.asarray(self.images, dtype=np.float32) / 255
                points = np.asarray(self.points, dtype=np.int32)
                masks = np.asarray(self.masks, dtype=np.float32)
                self.reset()
                yield inputs, points, masks


# In[9]:


def GAN_Main(result_dir="output", data_dir="data"):
    
    alpha = 0.0004
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)
    
    batchSize = 4
    Epochs = 150
    l1 = int(Epochs * 0.18)
    l2 = int(Epochs * 0.02)
    
    train_datagen = Data(data_dir, input_shape[:2], local_shape[:2])

    Gen = Generative(input_shape)
    Dis = Discriminative(input_shape, local_shape)
    
    optimizer = Adadelta()
#######
    orgVal = Input(shape=input_shape)
    mask = Input(shape=(input_shape[0], input_shape[1], 1))

    imgContent = Lambda(lambda x:x[0]*(1 - x[1]), output_shape=input_shape)([orgVal, mask])
    mimic = Gen(imgContent)
    completion = Lambda(lambda x:x[0]*x[2]+x[1]*(1 - x[2]), output_shape=input_shape)([mimic, orgVal, mask])
    Gen_container = Network([orgVal, mask], completion)
    Gen_out = Gen_container([orgVal, mask])
    Gen_model = Model([orgVal, mask], Gen_out)
    Gen_model.compile(loss='mse', optimizer=optimizer)
    
    inputLayer = Input(shape=(4,), dtype='int32')
    Dis_container = Network([orgVal, inputLayer], Dis([orgVal, inputLayer]))
    Dis_model = Model([orgVal, inputLayer], Dis_container([orgVal, inputLayer]))
    Dis_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    Dis_container.trainable = False
    totalModel = Model([orgVal, mask, inputLayer],
                      [Gen_out, Dis_container([Gen_out, inputLayer])])
    totalModel.compile(loss=['mse', 'binary_crossentropy'],
                      loss_weights=[1.0, alpha], optimizer=optimizer)
    
    for n in range(Epochs):
        progress = generic_utils.Progbar(len(train_datagen))
        for inputs, points, masks in train_datagen.flow(batchSize):
            Gen_image = Gen_model.predict([inputs, masks])
            real = np.ones((batchSize, 1))
            unreal = np.zeros((batchSize, 1))

            generatorLoss = 0.0
            discriminatorLoss = 0.0
            
            if n < l1:
                generatorLoss = Gen_model.train_on_batch([inputs, masks], inputs)
            else:
                discriminatorLoss_real = Dis_model.train_on_batch([inputs, points], real)
                discriminatorLoss_unreal = Dis_model.train_on_batch([Gen_image, points], unreal)
                discriminatorLoss = 0.5 * np.add(discriminatorLoss_real, discriminatorLoss_unreal)
                if n >= l1 + l2:
                    generatorLoss = totalModel.train_on_batch([inputs, masks, points],[inputs, real])
                    generatorLoss = generatorLoss[0] + alpha * generatorLoss[1]
            progress.add(inputs.shape[0])
        imgs = min(5,batchSize)
        Display, Axis = mplt.subplots(imgs, 3)

        Axis[0,0].set_title('Input Image')
        Axis[0,1].set_title('Output Image')
        Axis[0,2].set_title('Original Image')
        
        for i in range(imgs):
            Axis[i,0].imshow(inputs[i]*(1-masks[i]))
            Axis[i,0].axis('off')
            Axis[i,1].imshow(Gen_image[i])
            Axis[i,1].axis('off')
            Axis[i,2].imshow(inputs[i])
            Axis[i,2].axis('off')

        Display.savefig(os.path.join(result_dir,"Batch_%d.png"% n))
        mplt.close()

#Trained Model Files...        
    Gen.save(os.path.join(result_dir, "generator.h5"))
    Dis.save(os.path.join(result_dir, "discriminator.h5"))


# In[ ]:


def main():
    GAN_Main()

if __name__ == "__main__":
    main()


# In[ ]:




