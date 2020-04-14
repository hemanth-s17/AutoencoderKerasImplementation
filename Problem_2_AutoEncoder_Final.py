#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import os
import cv2


# In[8]:


labels= ['female', 'male', 'malestaff']
dataset = '/content/drive/My Drive/faces94/'
progress = 0
X_train = []
for label in labels:
    label_path = os.path.join(dataset, label)
    persons = os.listdir(label_path)
    for person in persons:
        person_path = os.path.join(label_path,person)
        images = os.listdir(person_path)
        for image in images:
            if(image.split('.')[-1] != 'jpg'):
                continue
            image = os.path.join(person_path,image)
            X_train.append(cv2.resize(cv2.imread(image),(128,128))/255)
        progress += 1
        print(progress)
        
X_train = np.stack(X_train)
print(X_train.shape)


# In[9]:


noise_factor = 0.5
x_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)


n = 10
plt.figure(figsize=(20,2))
for i in range(1,n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_train_noisy[i].reshape(128,128,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('ordata101.png')


# In[10]:


input_img = Input(shape=(128,128,3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(100, activation='relu')(x)
x = Dense(32*32*3)(x)
x = Reshape((32,32,3))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
encoder = Model(input_img,encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[11]:


history = autoencoder.fit(x_train_noisy, X_train, epochs=100, batch_size=128, shuffle=True, validation_split=0.2)


# In[13]:


n = 10
plt.figure(figsize=(20,2))
x_test_output = autoencoder.predict(x_train_noisy)
for i in range(1,n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_output[i].reshape(128,128,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('image11.png')


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[21]:


encoded_pred = encoder.predict(x_train_noisy)
print(encoded_pred.shape)
print(labels)
import csv
with open('Autoencoder_final.csv','w+') as fd:
  for i in range(3000):
    for j in encoded_pred[i]:
      fd.write(str(j))
      fd.write(",") 
    fd.write(str(labels[i])+";\n")


# In[ ]:




