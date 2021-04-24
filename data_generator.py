import numpy as np
import cv2 as cv
import os


class DataGenerator:
    def __init__(self, list_IDs, directory, batch_size=32, image_size=(256, 256), n_channels=3, shuffle=True):
        #Initialization
        self.list_IDs = list_IDs
        self.directory = directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        #Update indexes after each epoch.
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples.
        images = np.empty((self.batch_size, *self.image_size, self.n_channels))
        
        for i, ID in enumerate(list_IDs_temp):
            #Store image
            images[i, ] = (cv.imread(os.path.join(self.directory, ID)) - 127.5)/127.5
        
        return images
    
    def __len__(self):
        #Finds the number of batches per epoch.
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        #Generate Data
        images = self.__data_generation(list_IDs_temp)
        
        images = np.asarray(images)
        
        return images
    
# path_list = os.listdir("Data\\train")
# DG = DataGenerator(list_IDs = path_list, directory = 'Data\\train', batch_size = 16)
# print(DG.__getitem__(0).shape)