import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.preprocessing import LabelEncoder

datasets_path = "./CNC_Dataset/dataset.h5"
print(datasets_path)

labels_path = "./CNC_Dataset/labels.h5"
print(labels_path)

with h5py.File(labels_path, 'r') as h5file:
    labels = h5file['labels']
    
    Labels = labels[()]
    
print(Labels.shape)

with h5py.File(datasets_path, 'r') as h5file:
    dataset = h5file['vibration_data']
    
    Dataset = dataset[()]  # Load the entire dataset into a Numpy array

print(Dataset.shape)

Labels = Labels[:,0]
unique, counts = np.unique(Labels, return_counts=True)
percent = (counts*100)/len(Labels)
print(dict(zip(unique, np.round(percent, decimals=2))))
enc = LabelEncoder()
labels_num = enc.fit_transform(Labels)

# change this directory for your machine
root_dir = './'
Dataset_name = "CNC_Dataset"
# Create a folder for results
results_path = root_dir + "Results/" + Dataset_name
if os.path.exists(results_path):
    pass
else:
    try:
        os.makedirs(results_path)
    except:
        # in case another machine created the path meanwhile !:(
        pass
    
from Modules.ARSENAL_CV5 import ARSENAL
#Run The ARSENAL Module
ARSENAL(results_path, 
        Dataset_name,
        Dataset, 
        labels_num, 
        nb_folds=5, 
        num_kernels= 2000,
        n_estimators= 25,
        rocket_transform= "rocket",
        n_jobs=-1)
print(f"Working on {Dataset_name} finished successfully!")