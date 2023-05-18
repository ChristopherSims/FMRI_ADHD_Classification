import os,csv,warnings
import pandas as pd
from shutil import copy
import glob as gb
warnings.filterwarnings("ignore")

class FMRI_datasets():
    def __init__(self,dataset_directory,output_folder):
        self.Data_Directory = dataset_directory
        self.Datadirs = gb.glob(self.Data_Directory + "*")
        self.newDIR = output_folder
        if not os.path.exists(self.newDIR): ## Make dir if it isnt there
            os.makedirs(self.newDIR)
    def Make_FMRI_Dataset(self):
        dataset_tracker = {"Name":[],"DX":[],"Loc":[]} ### Dataset to be loaded to dict
        for folder in self.Datadirs:
            fmri_files = gb.glob(folder + "/*/*.gz")
            fmri_directory=gb.glob(folder + "/*/")
            pheno_csv_name = gb.glob(folder + "/*.csv") # Get phenotype data that is in every Dataset folder
            pheno = pd.read_csv(str(pheno_csv_name[0])) # read csv files
            folder_name = folder.replace(self.Data_Directory,"") # take out folder name from path
            for fmri_dir in fmri_directory:
                x = os.listdir(fmri_dir) 
                fmri_file = gb.glob(fmri_dir + "*.gz")
                if len(fmri_file) > 0:
                    file_str = fmri_file[0].replace(fmri_dir, "")
                    # Get Phenotype data for new folder
                    file_num = fmri_dir.replace(folder,"").replace("\\", "")
                    # Get Phenotype data for new folder
                    pheno_ind = pheno.index[pheno['ScanDir ID']==int(file_num)][0]
                    DX_value = pheno['DX'][pheno_ind]
                    if DX_value > 1: ## all ADHD is set to one for classification
                        DX_value = 1
                    ########## END
                    dataset_str = folder_name + "_" + file_str
                    extracted_dir = self.newDIR +"\\" + dataset_str
                    if not os.path.exists(extracted_dir): # if file doesnt exist
                        copy(fmri_file[0],extracted_dir)
                    dataset_tracker["Name"].append(dataset_str)
                    dataset_tracker["DX"].append(DX_value)
                    dataset_tracker["Loc"].append(extracted_dir)
                
        self.df = pd.DataFrame(data=dataset_tracker)
        self.df.to_csv(self.newDIR +"\\" + 'Total_Data_Pheno.csv', index=False)
        self.df.to_csv('Total_Data_Pheno.csv', index=False)
        return self.df
    def split_datasets(self,ratio):
        self.df = self.df.sample(frac=1)
        split = int(ratio*len(self.df))
        Training_df = self.df.iloc[0:split,:]
        Validation_df = self.df.iloc[split:len(self.df),:]
        ####
        Training_df.to_csv(newDIR +"\\" + 'Training_Data_Pheno.csv', index=False)
        Training_df.to_csv('Training_Data_Pheno.csv', index=False)
        ####
        Validation_df.to_csv(newDIR +"\\" + 'Validation_Data_Pheno.csv', index=False)
        Validation_df.to_csv('Validation_Data_Pheno.csv', index=False)
        return Training_df, Validation_df



##############################
# Create Training and Validation Sets
#########################

if __name__ == "__main__":
        Data_Directory = "FMRI_Data\\"
        newDIR = 'Extracted_FMRI_Data' ### folder to output to
        FMRI_Maker = FMRI_datasets(Data_Directory,newDIR)
        FMRI_Maker.Make_FMRI_Dataset()
        FMRI_Maker.split_datasets(0.5)
