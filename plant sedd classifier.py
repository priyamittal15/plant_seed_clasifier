#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import library
import pandas as pd
import numpy as np
import keras
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
import random
import pickle
from keras import models, layers, callbacks
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import shutil
import cv2
from math import sqrt, floor
from prettytable import PrettyTable


# In[2]:


classes= []
sample_counts= []

for f in os.listdir(r'D:\Plant seed classification Model\train'):
    train_class_path= os.path.join(r'D:\Plant seed classification Model\train', f)
    if os.path.isdir(train_class_path):
        classes.append(f)
        sample_counts.append(len(os.listdir(train_class_path)))

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
y_pos = np.arange(len(classes))

ax.barh(y_pos, sample_counts, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(classes)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Sample Counts')
ax.set_title('Sample Counts Per Class')

plt.show()


# In[3]:


#create validation set
def create_validation(validation_split=0.2):
        
    
    os.mkdir(r'D:\Plant seed classification Model\validation')
    for f in os.listdir(r'D:\Plant seed classification Model\train'):
        train_class_path= os.path.join(r'D:\Plant seed classification Model\train', f)
        if os.path.isdir(train_class_path):
            validation_class_path= os.path.join(r'D:\Plant seed classification Model\validation', f)
            os.mkdir(validation_class_path)
            files_to_move= int(0.2*len(os.listdir(train_class_path)))
            random_image= os.path.join(train_class_path, random.choice(os.listdir(train_class_path)))
            shutil.move(random_image, validation_class_path)
    print('Validation set created successfully using {:.2%} of training data'.format(validation_split))
    return 


# In[4]:


create_validation()


# In[ ]:


sample_counts= {}

for i, d in enumerate([r'D:\Plant seed classification Model\train', r'D:\Plant seed classification Model\validation']):

    classes= []
    sample_counts[d]= []

    for f in os.listdir(d):
        train_class_path= os.path.join(d, f)
        if os.path.isdir(train_class_path):
            classes.append(f)
            sample_counts[d].append(len(os.listdir(train_class_path)))

    #fig, ax= plt.subplot(221+i)
    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(classes))

    ax.barh(y_pos, sample_counts[d], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Sample Counts')
    ax.set_title('{} Sample Counts Per Class'.format(d.capitalize()))

plt.show()


# In[5]:


lower_bound= (24, 50, 0)
upper_bound= (55, 255, 255)

fig= plt.figure(figsize=(10, 10))
fig.suptitle('Random Pre-Processed Image From Each Class', fontsize=14, y=.92, horizontalalignment='center', weight='bold')

for i in range(12):
    sample_class=os.path.join(r'D:\Plant seed classification Model\test')   #preprocessing of images of test dataset
    random_image= os.path.join(sample_class, random.choice(os.listdir(sample_class)))
    img= cv2.imread(random_image)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, (150, 150))
    
    hsv_img= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    result = cv2.bitwise_and(img, img, mask=mask)

    fig.add_subplot(6, 4, i*2+1)
    plt.imshow(img)
    plt.axis('off')    

    fig.add_subplot(6, 4, i*2+2)
    plt.imshow(result)
    plt.axis('off')
    
plt.show()


# In[6]:


def color_segment_function(img_array):
    img_array= np.rint(img_array)
    img_array= img_array.astype('uint8')
    hsv_img= cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, (24, 50, 0), (55, 255, 255))
    result = cv2.bitwise_and(img_array, img_array, mask=mask)
    result= result.astype('float64')
    return result

#image function from keras.preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.4,1],
    rescale=1.0/255.0)

test_datagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function=color_segment_function)


# In[7]:


#divide our train and test folder into training and testing dataset.
training_set = train_datagen.flow_from_directory(r'D:\Plant seed classification Model\train',
                                                 target_size = (150, 150),
                                                 batch_size = 20,
                                                 class_mode = 'categorical')


# In[8]:


test_set = test_datagen.flow_from_directory(r'D:\Plant seed classification Model\validation',
                                            target_size = (150, 150),
                                            batch_size = 20,
                                            class_mode = 'categorical')


# In[9]:


# using neural network :
#Model Traning part begins
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))


model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))


model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.1))


model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.4))

model.add(layers.Dense(12, activation='softmax'))


# In[10]:


model.summary()


# In[11]:


best_cb= callbacks.ModelCheckpoint('model_best.h5', 
                                         monitor='val_loss', 
                                         verbose=1, 
                                         save_best_only=True, 
                                         save_weights_only=False, 
                                         mode='auto', 
                                         period=1)

opt= keras.optimizers.Adam(lr=0.0005, amsgrad=True)   # Learning rate=0.0005


# In[12]:


model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=30,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks= [best_cb])


# In[13]:


#load best model from training
model= models.load_model('model_best.h5')


# In[25]:


with open('model_history.pkl', 'wb') as f:
    pickle.dump(history, f)


# In[16]:


# plot the loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
#plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
#plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[18]:


pred= model.predict_generator(test_set, steps= test_set.n, verbose=1)


# In[26]:


from utils import label_map_util
from object_detection.utils import label_map_util
predicted_class_indices=np.argmax(pred,axis=1)

prediction_labels = [label_map_util[k] for k in predicted_class_indices]


# In[31]:


#import imutils
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# In[22]:


filenames= test_set.filenames


# In[24]:


#Final Result time for predicting the soln for some images dataset To check our model valid or not for predicting the results.
import csv
csvfile= open(r'D:\Plant seed classification Model\result_sample', 'w', newline='')
writer= csv.writer(csvfile)
headers= ['file', 'species']

writer.writerow(headers)
t = PrettyTable(headers)
for i, f, p in zip(range(len(filenames)), filenames, prediction_labels):
    writer.writerow([os.path.basename(f),p])
    if i <10:
        t.add_row([os.path.basename(f), p])
    elif i<13:
        t.add_row(['.', '.'])
csvfile.close()
print(t)


# In[ ]:




