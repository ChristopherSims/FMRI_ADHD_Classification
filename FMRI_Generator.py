import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow.keras as keras
from scipy.ndimage import zoom
import os
import nibabel as nib
from scipy import ndimage

'''
FMRI DATA Generator 
'''
class FMRIDataGenerator(keras.utils.Sequence):
    def __init__(self, datadict, batch_size):
        self.time_length = 30
        self.dim = [self.time_length, 28, 28, 28, 1] # [time, x, y, z, c]
        self.shuffle = True
        self.batch_size = batch_size
        self.data_len = len(np.arange(len(datadict['DX'].values())))
        self.labels = list(datadict['DX'].values())
        self.list_IDs = list(datadict['Loc'].values())
        self.indexes = np.arange(len(datadict['DX'].values()))
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_len / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        FMRI_data, FMRI_labels = self.__data_generation(indexes)

        return FMRI_data, FMRI_labels
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, index):
        x_data= np.empty((self.batch_size, *self.dim))
        x_labels = np.empty((self.batch_size), dtype=int)
        #for i, img_path in enumerate(list_IDs_temp):
        for i,idx in enumerate(index):
            x_data[i,] = self.preprocess_image(self.list_IDs[idx])
            x_labels[i] = self.labels[idx]
        return x_data,x_labels
    def preprocess_image(self, filepath):
        FMRI_data = nib.load(filepath).get_fdata()
        fmri_t = np.transpose(FMRI_data,axes=[3, 0, 1, 2])
        new_depth = 28
        new_width = 28
        new_height = 28
        old_depth = fmri_t.shape[1]
        old_width = fmri_t.shape[2]
        old_height = fmri_t.shape[3]
        ft = self.time_length/fmri_t.shape[0]
        fx = new_depth/old_depth
        fy = new_width/old_width
        fz = new_height/old_height
        fmri_resize = ndimage.zoom(fmri_t, (ft,fx,fy,fz) ,order=1)
        return fmri_resize[:,:,:,:,None]


'''
MRI Data Generator takes a dictonary as its input and returns the a set of MRI datasets whose 
size depends on the batch size

'''
class MRIDataGenerator(keras.utils.Sequence):
    def __init__(self, datadict, batch_size):
        self.dim = [64, 64, 64, 1] # [x, y, z, c]
        self.shuffle = False
        self.batch_size = batch_size
        self.data_len = len(np.arange(len(datadict['Loc'].values())))
        #self.labels = list(datadict['DX'].values())
        self.list_IDs = list(datadict['Loc'].values())
        self.indexes = np.arange(len(datadict['Loc'].values()))
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_len / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        MRI_data = self.__data_generation(indexes)
        return MRI_data
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, index):
        x_data= np.empty((self.batch_size, *self.dim))
        for i,idx in enumerate(index):
            x_data[i,] = self.preprocess_image(self.list_IDs[idx])
        return x_data, None ##unlabeled data 
    def preprocess_image(self, filepath):
        MRI_data = nib.load(filepath).get_fdata()
        MRI_t = np.transpose(MRI_data,axes=[0, 2, 1])
        new_depth = 64
        new_width = 64
        new_height = 64
        old_depth = MRI_t.shape[0]
        old_width = MRI_t.shape[1]
        old_height = MRI_t.shape[2]
        fx = new_depth/old_depth
        fy = new_width/old_width
        fz = new_height/old_height
        MRI_resize = ndimage.zoom(MRI_t , (fx,fy,fz) ,order=1)
        return MRI_resize[:,:,:,None] ## Add channel to output


'''
Multi Data Generator takes a dictonary as its input and returns the a set of MRI & FMRI datasets whose 
size depends on the batch size

The MRI dataset size must be equal to or greater then the FMRI dataset
'''
class MultiDataGenerator(keras.utils.Sequence):
    def __init__(self, FMRI_dict, MRI_dict, batch_size,MRI_input_shape,FMRI_input_shape):   
        self.dim_MRI = MRI_input_shape
        self.dim_FMRI = FMRI_input_shape # [time, x, y, z, c]
        self.time_length = FMRI_input_shape[0]
        self.batch_size = batch_size
        self.shuffle = True
        #FMRI
        self.data_len_FMRI = len(np.arange(len(FMRI_dict['DX'].values())))
        self.labels_FMRI = list(FMRI_dict['DX'].values())
        self.list_IDs_FMRI = list(FMRI_dict['Loc'].values())
        self.indexes = np.arange(len(FMRI_dict['DX'].values()))
        #MRI
        self.list_IDs_MRI = list(MRI_dict['Loc'].values())
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_len_FMRI / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        FMRI_data, MRI_data, FMRI_labels = self.__data_generation(indexes)
        return [FMRI_data, MRI_data], FMRI_labels ## Return two sets of data
        #return multi_data , multi_FMRI_labels
    def __data_generation(self, index):
        FMRI_data= np.empty((self.batch_size, *self.dim_FMRI))
        MRI_data= np.empty((self.batch_size, *self.dim_MRI))
        FMRI_labels = np.empty((self.batch_size), dtype=int)
        for i,idx in enumerate(index):
            FMRI_data[i,] = self.preprocess_image_FMRI(self.list_IDs_FMRI[idx])
            MRI_data[i,] = self.preprocess_image_MRI(self.list_IDs_MRI[idx])
            FMRI_labels[i] = self.labels_FMRI[idx]
        return  FMRI_data, MRI_data, FMRI_labels
    def on_epoch_end(self):
        #self.indexes = np.arange(len(self.list_IDs_FMRI))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def preprocess_image_MRI(self, filepath):
        MRI_data = nib.load(filepath).get_fdata()
        MRI_t = np.transpose(MRI_data,axes=[0, 2, 1])
        new_depth = self.dim_MRI[0]
        new_width = self.dim_MRI[1]
        new_height = self.dim_MRI[2]
        old_depth = MRI_t.shape[0]
        old_width = MRI_t.shape[1]
        old_height = MRI_t.shape[2]
        fx = new_depth/old_depth
        fy = new_width/old_width
        fz = new_height/old_height
        MRI_resize = ndimage.zoom(MRI_t , (fx,fy,fz) ,order=1)
        return MRI_resize[:,:,:,None]
    def preprocess_image_FMRI(self, filepath):
        FMRI_data = nib.load(filepath).get_fdata()
        fmri_t = np.transpose(FMRI_data,axes=[3, 0, 1, 2])
        new_depth = self.dim_FMRI[1]
        new_width = self.dim_FMRI[2]
        new_height = self.dim_FMRI[3]
        old_depth = fmri_t.shape[1]
        old_width = fmri_t.shape[2]
        old_height = fmri_t.shape[3]
        ft = self.time_length/fmri_t.shape[0]
        fx = new_depth/old_depth
        fy = new_width/old_width
        fz = new_height/old_height
        fmri_resize = ndimage.zoom(fmri_t, (ft,fx,fy,fz) ,order=1)
        return fmri_resize[:,:,:,:,None]
if __name__ == '__main__':
    import pandas as pd
    import tensorflow as tf
    # Trainingdict = pd.read_csv('Training_Data_Pheno.csv', index_col=False, squeeze=True).to_dict()
    # batch_size = 6
    # newgen = FMRIDataGenerator(Trainingdict, batch_size)
    # print(newgen.__len__())
    # xx,yy = newgen.__getitem__(1)
    # print(xx[1].shape)
    ##############
    # MRI Test
    ############
    MRIdict = pd.read_csv('Total_MRI.csv', index_col=False, squeeze=True).to_dict()
    # batch_size = 6
    # newgen_MRI = MRIDataGenerator(MRIdict, batch_size = 300)
    # #print(newgen_MRI.__len__())
    # dataset = tf.data.Dataset.from_tensor_slices(newgen_MRI.__getitem__(1))
    # print(type(dataset))
    #print(list(dataset.take(50)))
    #xx,yy = newgen_MRI.__getitem__(1)
    #print(xx.shape)

    MRIdict = pd.read_csv('Total_MRI.csv', index_col=False, squeeze=True).to_dict()
    Trainingdict = pd.read_csv('Training_Data_Pheno.csv', index_col=False, squeeze=True).to_dict()
    batch_size = 6
    newgen_Multi = MultiDataGenerator(Trainingdict,
                                    MRIdict,
                                    batch_size = 6,
                                    MRI_input_shape=(64,64,64,1),
                                    FMRI_input_shape=(30,28,28,28,1))
    print(newgen_Multi.__len__())
    newgen_Multi.on_epoch_end()
    xx,yy = newgen_Multi.__getitem__(36)
    print(type(xx))
    print(type(yy))
    print(xx[1].shape)

