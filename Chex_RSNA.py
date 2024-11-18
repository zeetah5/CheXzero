

import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import pandas as pd
from typing import List, Tuple, Optional
from data_process import img_to_hdf5
import sys
sys.path.append('../')  # Adjust the path as needed. Will probably need an adjustment.
from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval, load_clip  # Load clip must come from here.
from data_process import img_to_hdf5  # Import your data processing functions
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import torch
from torch.utils import data
import h5py
from pathlib import Path
# This ine is importnat and if often miss it apparently.


"""
Code for doing zero shot on the RSNA data with Chexpert. The overall structure for this is sligtly differenct because now do we

    - I a

    - Craeating the repository with all the images and a label file that describes is all. 

    - Need to create a h5 that organizes everything. How does it work for the COVID data which is divided into 4 directories which hold our data.

"""

#---------------------Data prepearation functions--------------------------------------

# Skipped over creating the directory from the DICOM files 

# Keeping the creation of label file even thoug it should really only be called once
def create_label_file(original_csv, output_csv, image_dir):
    # Load the original CSV file
    df = pd.read_csv(original_csv)

    # Create the imgpath column
    df['imgpath'] = df['patientId'].apply(lambda x: os.path.join(image_dir, f'{x}.jpg'))

    # Map Target to No Finding and Pneumonia columns
    df['No Finding'] = df['Target'].apply(lambda x: 1 if x == 0 else 0)
    df['Pneumonia'] = df['Target'].apply(lambda x: 1 if x == 1 else 0)

    # Remove duplicates
    df = df.drop_duplicates(subset='patientId')

    # Keep only necessary columns
    df = df[['imgpath', 'No Finding', 'Pneumonia']]

    # Save the new CSV file
    df.to_csv(output_csv, index=False)


#-----------------Preprocessing----------------------------------------------

"""
Getting the HDF5 we need. I hope this preprocessing will work, but we will see.
"""
def preprocess_rsna(csv_path, hdf5_out_path):
    df = pd.read_csv(csv_path)
    # we want to turn th entire direcrory in 
    cxr_paths = df['imgpath'].tolist()
    # lets hope this work.
    img_to_hdf5(cxr_paths, hdf5_out_path, resolution=320)


"""
Make true labels. Cutlabels are used
The difference between this one and the original is that: 
"""
def make_true_labels(
        cxr_true_lables_path:str,
        cxr_labels: List[str],
        cutlabels: bool = False
):
    # Create ground truth labels
    full_labels = pd.read_csv(cxr_true_lables_path)
    if cutlabels:
        full_labels = full_labels.loc[:, cxr_labels]
    else:
        # Assuming the csv has 'imgpath' as the first column.
        full_labels = full_labels.drop(columns=['imgpath'])
    y_true = full_labels.to_numpy()

    return y_true


#-------------The code itself-------------------------------------------

"""
The make function makes the model, the data loader and the ground truth label. 
It does not seem if there are any differences between this one and the original. hmmm.
"""
def make(model_path:str, cxr_filepath:str, pretrained:bool = True, context_length: int = 77):

    # Load the model.
    model = load_clip(
        model_path= model_path,
        pretrained=pretrained,
        context_length=context_length
    )

    # load data
    transformations = [
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    if pretrained:
        input_resolution = 224
        transformations.append(Resize(input_resolution))
    transform = Compose(transformations)

    # Creatning the dataset.
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)

    return model, loader



"""
This is a function for handling the chest x-ray dataset. 
We did not have to make any modifications to it.

"""
class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample



"""
Finally will we make some modifications to the ensemble models function. 
"""

def ensemble_models( model_paths: List[str],  cxr_filepath: str,  cxr_labels: List[str],  cxr_pair_template: Tuple[str],  cache_dir: str = None,  save_name: str = None,) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """
    predictions = []
    model_paths = sorted(model_paths)
    for path in model_paths:
        model_name = Path(path).stem # returns the name without the .pt part of the name.

        # load in model and 
        model, loader = make(
            model_path=path,
            cxr_filepath=cxr_filepath
        )

        # Path to the cached prediction.
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # If the prediction is already cached, dont recompute the prediction. 
        if cache_dir is not None and os.path.exists(cache_path):
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else:
            # Cached prediction not found. Compute the preds.
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)

    """
    Compute the average prediction, important to note that it is the average since we have several 
    models and we ensamble them. 
    """
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg




"""
I thought this would work, but I think we need some kind of preprocessing which is not happening here. 
To do that do we need the other parts. 
"""
def main():

    # Creat

    # the filepath to the HDF5 file we will create
    cxr_filepath: str = 'data/rsna_images.h5'
    # this should be fairly easy to make. Optinal means that it must have type str or none.
    cxr_true_labels_path :Optional[str] = 'data/output_labels.csv'

    model_dir: str = './checkpoints/chexzero_weights'  # Where the pretrained weights are
    predictions_dir: Path = Path('data/predictions')  # Where to save predictions
    cache_dir: Path = predictions_dir / "cached"  # Where to cache ensembled predictions. Might not need to cache.

    context_length: int = 77

    cxr_labels= ['No finding', 'Pneuomnia'] 
    cxr_pair_template = ("{}", "no {}")

    # Where the label file is stored. 
    label_file_path = '../rsna_labels.csv'

    # create the csv file. This is not necessary for RSNA in chexpert since it will have the same as the others.
    #create_label_file(original_csv, output_csv, jpeg_dir)

    # will not have to do the preprocessing 
    #preprocess_rsna(label_file_path, cxr_filepath)

    # The model paths 
    model_paths = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt'): # adjust based on model file extension 
                full_dir = os.path.join(subdir, file)
                model_paths.append(full_dir)

    print("Model paths:", model_paths)


      
     #Predictions. Will use ensemble models.
     # should maybe start by only using one model. 
    predictions, y_pred_avg = ensemble_models(
        model_paths= [ './checkpoints/chexzero_weights\\best_64_5e-05_original_18000_0.862.pt'],  # 
        cxr_filepath=cxr_filepath,
        cxr_labels=cxr_labels,
        cxr_pair_template=cxr_pair_template,
        cache_dir=cache_dir
    )
   
    # Save averaged predictions
    pred_name = 'rsna_preds.npy'
    predictions_dir = predictions_dir / pred_name
    np.save(file = predictions_dir, arr = y_pred_avg)

    # Get ground truth labels. Need to look thoroughly at how it's done 
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels, cutlabels=False)

    # Evaluate the model
    cxr_results = evaluate(y_pred_avg, test_true, cxr_labels)

    # Bootstrap results for confidence intervals
    bootstrap_results = bootstrap(y_pred_avg, test_true, cxr_labels)

    # Display AUC with confidence intervals
    print(bootstrap_results[1])

    

if __name__ == '__main__':  
    main()

