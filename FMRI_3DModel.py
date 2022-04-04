
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, MaxPool3D, TimeDistributed, Flatten, LSTM, Dense
from tensorflow.keras.layers import BatchNormalization,GlobalAveragePooling3D
from tensorflow.keras.layers import Dropout,GRU,concatenate
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger
from keras.utils.vis_utils import plot_model
from FMRI_Generator import FMRIDataGenerator,MRIDataGenerator,MultiDataGenerator
import pandas as pd
from datetime import datetime
import logging
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.FATAL)
########################
# DEBUG GPU issues
#########################
#tf.config.list_physical_devices()
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)


class FMRI_3D_Model():
    def __init__(self,FMRI_Shape,MRI_Shape,train_dic_loc,val_dic_loc,mri_dic_loc):
        self.input_shape_FMRI = FMRI_Shape #(30,28,28,28,1)
        self.input_shape_MRI = MRI_Shape #(64,64,64,1)
        self.MRI_dict = pd.read_csv(mri_dic_loc, index_col=False, squeeze=True).to_dict()
        self.Training_dict = pd.read_csv(train_dic_loc, index_col=False, squeeze=True).to_dict()
        self.Validation_dict = pd.read_csv(val_dic_loc, index_col=False, squeeze=True).to_dict()
        self.partition = {'train': self.Training_dict, 
                    'validation': self.Validation_dict}

    def get_fmri_model_LSTM(self,input_shape):
        """Build a 3D convolutional neural network model."""
        inputs = keras.Input(input_shape)
        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_1")(inputs)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_1")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_2")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_2")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=128, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_3")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_3")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=256, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_4")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_4")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Flatten(), name="Flatten_Layer2")(x)
        x = LSTM(1, dropout = 0.3, recurrent_dropout = 0.3, name="LSTM_Layer")(x)
        #x = Dense(512, activation="relu")(x)
        #x = TimeDistributed(GlobalAveragePooling3D())(x)
        

        #x = TimeDistributed(Flatten(), name="Flatten_Layer")(x)
        #x = Dropout(0.3)(x)

        outputs = Dense(units=1, activation='relu')(x)

        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn")
        return model
    def get_2step_model_LSTM(self,FMRI_input_shape,MRI_input_shape):
        """Build a 3D convolutional neural network model."""
        inputA = keras.Input(FMRI_input_shape)
        inputB = keras.Input(MRI_input_shape)
        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_1")(inputA)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_1")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_2")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_2")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=128, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_3")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_3")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=256, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_4")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_4")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Flatten(), name="Flatten_Layer1")(x)
        x = LSTM(512, dropout = 0.3, recurrent_dropout = 0.3, name="LSTM_Layer")(x)

        outputsA = Dense(units=512, activation='sigmoid')(x)
        x = keras.Model(inputs = inputA, outputs = outputsA)
        #################################################
        # MRI Part
        ################################################
        y = Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer_1")(inputB)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool_1")(y)
        y = BatchNormalization()(y)

        y = Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer2")(y)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool_2")(y)
        y = BatchNormalization()(y)

        y = Conv3D(filters=128, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer3")(y)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool_3")(y)
        y = BatchNormalization()(y)

        y = Conv3D(filters=256, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer4")(y)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool4")(y)
        y = BatchNormalization()(y)

        y = Flatten(name="MRI_Flatten_Layer1")(y)
        outputsB = Dense(units=512, activation='sigmoid')(y)
        y = keras.Model(inputs=inputB, outputs=outputsB )

        ##############################
        # Combine both models
        ##############################
        combined = concatenate([x.output, y.output])

        outputC = Dense(units=1, activation="sigmoid")(combined)
        # Define the model.
        model = keras.Model(inputs = [x.input, y.input], outputs = outputC, name="MRI_FMRI_3D_cnn")
        return model

    def get_fmri_model_GRU(self,input_shape):
        """Build a 3D convolutional neural network model."""
        inputs = keras.Input(input_shape)

        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_1")(inputs)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_1")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_2")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_2")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=128, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_3")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_3")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=256, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_4")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_4")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Flatten(), name="Flatten_Layer_1")(x)
        x = GRU(1, dropout = 0.3, recurrent_dropout = 0.3, name="GRU_Layer")(x)

        outputs = Dense(units=1, activation='sigmoid')(x)

        # Define the model.
        model = keras.Model(inputs, outputs, name="FMRI_GRU_3Dcnn")
        return model

    def get_2step_model_GRU(self,FMRI_input_shape,MRI_input_shape):
        """Build a 3D convolutional neural network model."""
        inputA = keras.Input(FMRI_input_shape)
        inputB = keras.Input(MRI_input_shape)
        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_1")(inputA)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_1")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_2")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_2")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=128, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_3")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_3")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Conv3D(filters=256, kernel_size=(2,2,2), activation="relu",padding = 'same'), name="Conv_Layer_4")(x)
        x = TimeDistributed(MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2)), name="Max_pool_4")(x)
        x = BatchNormalization()(x)

        x = TimeDistributed(Flatten(), name="Flatten_Layer1")(x)
        x = GRU(512, dropout = 0.3, recurrent_dropout = 0.3, name="LSTM_Layer")(x)

        outputsA = Dense(units=512, activation='sigmoid')(x)
        x = keras.Model(inputs = inputA, outputs = outputsA)
        #################################################
        # MRI Part
        ################################################
        y = Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer_1")(inputB)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool_1")(y)
        y = BatchNormalization()(y)

        y = Conv3D(filters=64, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer2")(y)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool_2")(y)
        y = BatchNormalization()(y)

        y = Conv3D(filters=128, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer3")(y)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool_3")(y)
        y = BatchNormalization()(y)

        y = Conv3D(filters=256, kernel_size=(2,2,2), activation="relu",padding = 'same', name="MRI_Conv_Layer4")(y)
        y = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), name="MRI_Max_Pool4")(y)
        y = BatchNormalization()(y)

        y = Flatten(name="MRI_Flatten_Layer1")(y)
        outputsB = Dense(units=512, activation='sigmoid')(y)
        y = keras.Model(inputs=inputB, outputs=outputsB )

        ##############################
        # Combine both models
        ##############################
        combined = concatenate([x.output, y.output])

        outputC = Dense(units=1, activation="sigmoid")(combined)
        # Define the model.
        model = keras.Model(inputs = [x.input, y.input], outputs = outputC, name="MRI_FMRI_3D_cnn")
        return model

    def run_fmri_model_LSTM(self,batch_size,epochs):
        model = self.get_fmri_model_LSTM(self.input_shape_FMRI)
        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        training_generator = FMRIDataGenerator(self.partition['train'], batch_size)
        validation_generator = FMRIDataGenerator(self.partition['validation'], batch_size)
        #############################
        curr_time = f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
        logger_path = "log/FMRI_RUN_{time}_SM_LSTM.csv".format(time=curr_time)
        csv_logger = CSVLogger(logger_path, append=True)
        callbacks = [csv_logger]
        #####################################
        model.fit(
                x=training_generator,
                verbose=1, 
                callbacks=callbacks,
                validation_data=validation_generator,
                epochs=epochs,
                use_multiprocessing=True,
                workers=3)
    def run_fmri_model_GRU(self,batch_size,epochs):
        model = self.get_fmri_model_GRU(self.input_shape_FMRI)
        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        training_generator = FMRIDataGenerator(self.partition['train'], batch_size)
        validation_generator = FMRIDataGenerator(self.partition['validation'], batch_size)
        #############################
        curr_time = f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
        logger_path = "log/FMRI_RUN_{time}_SM_GRU.csv".format(time=curr_time)
        csv_logger = CSVLogger(logger_path, append=True)
        callbacks = [csv_logger]
        #####################################
        model.fit(
                x=training_generator,
                verbose=1, 
                callbacks=callbacks,
                validation_data=validation_generator,
                epochs=epochs,
                use_multiprocessing=True,
                workers=3)

    def run_MM_FMRI_LSTM(self,batch_size,epochs):
        model = self.get_2step_model_LSTM(self.input_shape_FMRI,self.input_shape_MRI)
        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        TT = MultiDataGenerator(FMRI_dict=self.partition['train'],
                                MRI_dict= self.MRI_dict, 
                                batch_size = batch_size,
                                MRI_input_shape=self.input_shape_MRI,
                                FMRI_input_shape=self.input_shape_FMRI)
        VV = MultiDataGenerator(FMRI_dict=self.partition['validation'],
                                MRI_dict=self.MRI_dict, 
                                batch_size = batch_size,
                                MRI_input_shape=self.input_shape_MRI,
                                FMRI_input_shape=self.input_shape_FMRI)
        #############################
        curr_time = f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
        logger_path = "log/FMRI_RUN_{time}_MM_LSTM.csv".format(time=curr_time)
        csv_logger = CSVLogger(logger_path, append=True)
        callbacks = [csv_logger]
        #####################################
        model.fit(
                x=TT,
                verbose=1, 
                callbacks=callbacks,
                validation_data=VV,
                epochs=epochs,
                use_multiprocessing=True,
                workers=3)

    def run_MM_FMRI_GRU(self,batch_size,epochs):
        model = self.get_2step_model_GRU(self.input_shape_FMRI,self.input_shape_MRI)
        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        TT = MultiDataGenerator(FMRI_dict=self.partition['train'],
                                MRI_dict= self.MRI_dict, 
                                batch_size = batch_size,
                                MRI_input_shape=self.input_shape_MRI,
                                FMRI_input_shape=self.input_shape_FMRI)
        VV = MultiDataGenerator(FMRI_dict=self.partition['validation'],
                                MRI_dict=self.MRI_dict, 
                                batch_size = batch_size,
                                MRI_input_shape=self.input_shape_MRI,
                                FMRI_input_shape=self.input_shape_FMRI)
        #############################
        curr_time = f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
        logger_path = "log/FMRI_RUN_{time}_MM_GRU.csv".format(time=curr_time)
        csv_logger = CSVLogger(logger_path, append=True)
        callbacks = [csv_logger]
        #####################################
        model.fit(
                x=TT,
                verbose=1, 
                callbacks=callbacks,
                validation_data=VV,
                epochs=epochs,
                use_multiprocessing=True,
                workers=3)







if __name__ == '__main__':
    epochs = 10
    batch_size = 3
    input_shape_FMRI=(30,28,28,28,1)
    input_shape_MRI=(64,64,64,1)
    FMRI_3DCNN = FMRI_3D_Model(FMRI_Shape=input_shape_FMRI,
                        MRI_Shape=input_shape_MRI,
                        train_dic_loc='Training_Data_Pheno.csv',
                        val_dic_loc='Validation_Data_Pheno.csv',
                        mri_dic_loc='Total_MRI.csv')
    #FMRI_3DCNN.run_fmri_model_GRU(batch_size=batch_size,epochs=epochs)
    #FMRI_3DCNN.run_fmri_model_LSTM(batch_size=batch_size,epochs=epochs)
    #FMRI_3DCNN.run_MM_FMRI_GRU(batch_size=batch_size,epochs=epochs)
    FMRI_3DCNN.run_MM_FMRI_LSTM(batch_size=batch_size,epochs=epochs)