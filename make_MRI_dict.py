import warnings,os,random
import glob as gb
import pandas as pd
from shutil import copy
warnings.filterwarnings("ignore")




class MRI_dataset():
    def __init__(self,olddir,newdir):
        self.old_data_dir = olddir + "/"
        self.newDIR = 'Extracted_MRI_Data'
        if not os.path.exists(self.newDIR):
            os.makedirs(self.newDIR)
    
    
    def Make_MRI_Dataset(self):
        dataset_tracker = {"Name":[],"Loc":[]}
        MRI_file_list = gb.glob(self.old_data_dir + "*.gz")
        for MRI_file in MRI_file_list:
            file_name = MRI_file.replace(self.old_data_dir,"").replace("\\", "")
            extracted_loc = self.newDIR +"\\" + file_name
            if not os.path.exists(extracted_loc):
                copy(MRI_file,extracted_loc)
            dataset_tracker["Name"].append(file_name)
            dataset_tracker["Loc"].append(extracted_loc)

        self.df = pd.DataFrame(data=dataset_tracker)
        self.df.to_csv(self.newDIR +"\\" + 'Total_MRI.csv', index=False)
        self.df.to_csv('Total_MRI.csv', index=False)
        return self.df


if __name__ == "__main__":
    old_data_dir = "MRI_Data"
    newDIR = 'Extracted_MRI_Data'
    MRID = MRI_dataset(olddir= old_data_dir,
                    newdir=newDIR)
    MRID.Make_MRI_Dataset()