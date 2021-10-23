# Data Generator
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate, shift
from PIL import Image

class ImageDataLoader(tf.keras.utils.Sequence):
    def __init__(self,
                 df,
                 batch_size=32,
                 dim=(224,224, 3),
                 img_size = (224,224),
                 n_actions =21,
                 n_action_cls = 5,
                 data_mean=0,
                 data_std=1,
                 data_prefix='',
                 shuffle=True,
                 Augment=True):
        
        'Initialization'
        self.dim = dim  # Dimensions of the input
        self.batch_size = batch_size # Batch size
        self.img_size = img_size # Image size
        self.n_actions = n_actions  # Number of actions
        self.n_action_cls = n_action_cls  # Number of action class   
        self.shuffle = shuffle  # Flag to shuffle data at the end of epoch
        self.Augment = Augment  # Flag to augment the data

        # The data is input as a pandas dataframe, we need to read the relevent fields
        self.df = df
        self.action = df['action_label'].values.tolist()
        self.action_class = df['action_class_label'].values.tolist()
        self.image_ids = np.arange(len(self.df.index)).tolist()
        self.data_prefix = data_prefix
        
        # Data normalization parameters
        self.data_mean = data_mean
        self.data_std = data_std
        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data for the given index'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        data_ids_temp = [self.image_ids[k] for k in indexes]
        action_temp = [self.action[k] for k in indexes]
        action_class_temp = [self.action_class[k] for k in indexes]
        
        # Generate data
        X, first_target, second_target = self.__data_generation(data_ids_temp, action_temp, action_class_temp)
        
        return X,[first_target, second_target]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, data_ids_temp, action_temp, action_class_temp):
        'Generates data containing batch_size samples' 

        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        first_target = np.empty((self.batch_size), dtype=int)
        second_target = np.empty((self.batch_size), dtype = int)
        
        # Generate data
        for i, ids in enumerate(data_ids_temp):
            
            X[i,] = self.__read_data_instance(data_ids_temp[i])
            first_target[i] = action_temp[i]
            second_target[i] = action_class_temp[i]
            
        return X, first_target, second_target


    def __read_data_instance(self, pid):
      # Read an image
      filepath = self.data_prefix + self.df.iloc[pid]['FileName']
      
      data = Image.open(filepath)
      data = data.resize(self.img_size) # Resize image
      data = np.asarray(data) # Convert into numpy array
      
      #Augmentation techniques
      if self.Augment:
        data = tf.image.resize_with_crop_or_pad(data, 240, 240)
        data = tf.image.random_crop(data, size = [224,224,3])
        data = tf.image.random_flip_left_right(data)
        data = tf.image.random_brightness(data, max_delta = 0.5)
        rot = np.random.rand(1) < 0.5
        if rot:
            rot = np.random.randint(-15,15, size=1)
            data = rotate(data, angle=rot[0], reshape=False)
          
        shift_val = np.random.randint(-5, high=5, size=2, dtype=int).tolist() + [0,]
        data = shift(data, shift_val, order=0, mode='constant', cval=0.0, prefilter=False)

      X = data
      # Input normalization
      X = (X - self.data_mean)/self.data_std
      return X

# Data Generator
class TestDataLoader(tf.keras.utils.Sequence):
    def __init__(self,
                 df,
                 batch_size=32,
                 dim=(224,224, 3),
                 img_size = (224,224),
                 data_mean=0,
                 data_std=1,
                 data_prefix=''):
        
        'Initialization'
        self.dim = dim  # Dimensions of the input
        self.batch_size = batch_size # Batch size
        self.img_size = img_size # Image size

        # The data is input as a pandas dataframe, we need to read the relevent fields
        self.df = df
        self.image_ids = np.arange(len(self.df.index)).tolist()
        self.data_prefix = data_prefix
        self.indexes = np.arange(len(self.image_ids))
        # Data normalization parameters
        self.data_mean = data_mean
        self.data_std = data_std
        
        #self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data for the given index'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        data_ids_temp = [self.image_ids[k] for k in indexes]
        
        # Generate data
        X = self.__data_generation(data_ids_temp)
        
        return X

            
    def __data_generation(self, data_ids_temp):
        'Generates data containing batch_size samples' 

        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        
        # Generate data
        for i, ids in enumerate(data_ids_temp):
            X[i,] = self.__read_data_instance(data_ids_temp[i])
            
        return X


    def __read_data_instance(self, pid):
      # Read an image
      filepath = self.data_prefix + self.df.iloc[pid]['FileName']
      
      data = Image.open(filepath).convert('RGB')
      data = data.resize(self.img_size) # Resize image
      data = np.asarray(data) # Convert into numpy array
      X = data
      # Input normalization
      X = (X - self.data_mean)/self.data_std
      return X