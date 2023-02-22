See https://arxiv.org/abs/2205.11993 for details on performance


To install packages run
!pip install -r requirements.txt


In order to generate 3D GAN images:
1. go to https://github.com/ChristopherSims/3DStyleGAN ; for information about the program
2. Run 3DGan.ipyb on google colab ; in order to run on local machines you will need several additional packages that are non-intuitive to install


Data Generation:
Extracted FMRI data must be in FMRI_data with the datasets seperated into thier respective folders, you can include
any datasets you want as long as they have the phonotype.csv inside of them
Once the zipped folders are extracted into FMRI_Data:
1. Run Make_Data_FMRI, this will read phonotype data and load it into Exctracted_FMRI_Data and create a dict of values for training
2. Take 3D GAN MRI images and put them in MRI_Data and run Make_MRI_Dict.py, this will put data in Extracted_MRI_Data and create a dict for
the dataloader

Run Training:
1. Read FMRI_Generator to understand how the dataloading works if desired
2. Run FMRI_3DModel.py to train network; by default Multi modal LSTM is selected, uncomment the code to run the other models

3. FMRI_3DCNN.run_fmri_model_GRU(batch_size=batch_size,epochs=epochs)
4. FMRI_3DCNN.run_fmri_model_LSTM(batch_size=batch_size,epochs=epochs)
5. FMRI_3DCNN.run_MM_FMRI_GRU(batch_size=batch_size,epochs=epochs)
6. FMRI_3DCNN.run_MM_FMRI_LSTM(batch_size=batch_size,epochs=epochs)
