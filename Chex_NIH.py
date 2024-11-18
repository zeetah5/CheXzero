from itertools import chain

import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple, Optional
import sys
sys.path.append('../')  # Adjust the path as needed. Will probably need an adjustment.
from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval, load_clip  # Load clip must come from here.
from data_process import img_to_hdf5  # Import your data processing functions
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor
import torch
from torch.utils import data
import h5py
from pathlib import Path
import clip
from model import CLIP



"""
Code for handling the NIH_data with the CheXpert model. 
I add the paths here to make it better.
"""
def create_label_file(csv_path, image_dir):
    df = pd.read_csv(csv_path)
   # Get a list of all unque labels.
    all_labels = df['Finding Labels'].unique()
    # Split the labels by "|", flatten the list and get unique labels.

    # Since all labels will only return unique permutations.    
    unique_labels = set(chain(*[labels.split('|') for labels in all_labels]))
    unique_labels = sorted(unique_labels)

    #  Now need to do a one_hot encoding for the labels.
    #  When we write df[label] do we crate those labels.
    for label in unique_labels:
        df[label] = 0

    # Fill in the one hot encoding as per instructions.
    for index, row  in df.iterrows():
        labels = row['Finding Labels'].split('|') # finding labels is the labels they hav 
        for label in labels:
            df.at[index, label] = 1


    # Now do we only need to keeo the images that we actually need. Store the images in the NIH_IMAGES directory. 

    #Create the image_path column which is always needed. 
    df['imgpath'] = df['Image Index'].apply(lambda x: os.path.join(image_dir, x))

    # we also only need to only keep imgpath and label columns.
    columns_to_keep = ['imgpath'] + unique_labels
    df = df[columns_to_keep]


    return df, unique_labels


"""
Time to do some preprocessing. 
"""

def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    return new_img

"""
This is mostly the same as previously, but now is it adapted to accecpted a dataframe directly.
It does a lot of the preprocessing for us as well. 
"""
def img_and_labels_to_hdf5(df, hdf5_filepath, resolution=320):
    imgpaths = df['imgpath'].tolist()
    labels = df.drop(columns=['imgpath']).values.astype(np.float32)  # One-hot labels

    num_images = len(imgpaths)
    num_classes = labels.shape[1]

    failed_images = []
    with h5py.File(hdf5_filepath, 'w') as h5f:
        img_dset = h5f.create_dataset('cxr', shape=(num_images, resolution, resolution), dtype=np.uint8)
        label_dset = h5f.create_dataset('labels', data=labels)
        for idx, path in enumerate(tqdm(imgpaths, desc="Processing images")):
            try:
                img = Image.open(path).convert('L')  # Convert to grayscale
                img = preprocess(img, desired_size=resolution)
                img_array = np.array(img, dtype=np.uint8)
                img_dset[idx] = img_array
            except Exception as e:
                failed_images.append((path, e))
        print(f"{len(failed_images)} / {len(imgpaths)} images failed to be added to HDF5.")
        if failed_images:
            print("Failed images:", failed_images)


"""
We will now change the dataset to handle the NIH dataset. I do not know if this is correct, honestly. 

However, if we want to make this more closer to how it 
"""
class NIHDataset(data.Dataset):
    def __init__(
        self,
        img_path: str,
        transform=None,
    ):
        super().__init__()
        self.img_path = img_path
        self.transform = transform

        # Get the number of samples
        with h5py.File(self.img_path, 'r') as h5_file:
            self.num_samples = h5_file['cxr'].shape[0]

        self.h5_file = None  # Will be initialized per worker

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.img_path, 'r')

        img = self.h5_file['cxr'][idx]
        label = self.h5_file['labels'][idx]

        img = Image.fromarray(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.float32)
        sample = {'img': img, 'label': label}
        return sample

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

"""
Adjusting the make true labels. 
I am fairly confident that it is the same one. 
"""
def make_true_labels(
    cxr_true_labels_path: str,
    cxr_labels: List[str],
    cutlabels: bool = False
):
    # Create the ground truth labels.
    full_labels = pd.read_csv(cxr_true_labels_path)
    if cutlabels:
        full_labels = full_labels.loc[:, cxr_labels]
    else:
        # Assuming the csv has 'imgpath' as the first column
        full_labels = full_labels.drop(columns=['imgpath'])
    y_true = full_labels.to_numpy()
    return y_true

"""
Time for defining the make. I use the provided one here. The important detail here is that ToTensor is used here, but I do not know if this is necessary.
"""
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor

def make(
    model_path: str,
    cxr_filepath: str,
    pretrained: bool = True,
    context_length: int = 77,
):
    # Load model
    model = load_clip(
        model_path=model_path,
        pretrained=pretrained,
        context_length=context_length
    )

    # Define transformations
    transformations = []
    if pretrained:
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transformations.append(ToTensor())
    transformations.append(Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))
    transform = Compose(transformations)

    # Create dataset
    torch_dset = NIHDataset(
        img_path=cxr_filepath,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(torch_dset, batch_size=32, shuffle=False, num_workers=0)

    return model, loader

"""
Slightly adjust this from the COVID by ensuring that the model is known completely.
"""
def zeroshot_classifier(classnames, templates, model, context_length=77):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts, context_length=context_length).to(next(model.parameters()).device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

"""
The predict function as well. We should not use sigmoid. We should in fact use softmax and not sigmoid.
"""
def predict(loader, model, zeroshot_weights):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running Inference"):
            images = batch['img'].to(model.device)
            labels = batch['label']
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ zeroshot_weights
            probs = logits.softmax(dim=-1)
            y_pred.append(probs.cpu().numpy())
            y_true.append(labels.numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_pred, y_true

"""
Should be the same as previously. 
"""
def ensemble_models(model_paths: List[str], cxr_filepath: str, cxr_labels: List[str], templates: List[str], cache_dir: str = None, save_name: str = None,) -> Tuple[List[np.ndarray], np.ndarray]:
    predictions = []
    model_paths = sorted(model_paths)
    for path in model_paths:
        model_name = Path(path).stem
        # Load model and data loader
        model, loader = make(
            model_path=path,
            cxr_filepath=cxr_filepath
        )

        # Create zeroshot classifier
        zeroshot_weights = zeroshot_classifier(cxr_labels, templates, model)

        # Predict
        y_pred, _ = predict(loader, model, zeroshot_weights)

        predictions.append(y_pred)

    # Compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg


"""
Evaluate. It is a per label evaluation.
"""
def evaluate(y_pred, y_true, cxr_labels):
    from sklearn.metrics import roc_auc_score, f1_score

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    # Compute ROC AUC and F1 score for each class
    roc_auc = []
    f1_scores = []
    for idx, class_name in enumerate(cxr_labels):
        try:
            if np.unique(y_true[:, idx]).size > 1:
                auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
                roc_auc.append(auc)
            else:
                roc_auc.append(float('nan'))
        except ValueError:
            roc_auc.append(float('nan'))
        preds = (y_pred[:, idx] > 0.5).astype(np.float32)
        f1 = f1_score(y_true[:, idx], preds, zero_division=0)
        f1_scores.append(f1)
        print(f"Class: {class_name}, ROC AUC: {roc_auc[-1]:.4f}, F1 Score: {f1:.4f}")

    # If you need to return results
    results = {'roc_auc': roc_auc, 'f1_scores': f1_scores}
    return results


"""
Finally time for the main function. This contains versions off effectively  every function that is needed. We will se nho
"""
def main():

    """
    All of the paths. Will need to adjust these paths. 
    """
    csv_path = '/path/to/Data_Entry_2017.csv'  # Update this path
    image_dir = '/path/to/images'  # Update this path
    hdf5_out_path = 'nih_data.h5'
    model_dir = '/path/to/chexzero_weights'  # Where the pretrained weights are
    predictions_dir = Path('/path/to/output/predictions')
    cache_dir = predictions_dir / "cached"

    context_length = 77

    """
    Loading the data. Important to note that the labels come from here so no need to define them separetly. 
    """
    df, cxr_labels = create_label_file(csv_path, image_dir)

    # Create the HDF5 file. Unlike, perhaps, the others will be used the one returned from create_label_file
    img_and_labels_to_hdf5(df, hdf5_out_path)

    # Get the model paths. 
    model_paths = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                full_dir = os.path.join(subdir, file)
                model_paths.append(full_dir)
    print("Model paths:", model_paths)


    # Templates for NIH. The template is wrong.  It should have two parts. 
    templates = ["This chest X-ray shows signs of {}."]

    # Run the ensemble models. 
    predictions, y_pred_avg  = ensemble_models(
        model_paths=model_paths,
        cxr_filepath=hdf5_out_path,
        cxr_labels=cxr_labels,
        templates=templates, 
        cache_dir=cache_dir
    )

    # Save the average predictions. 
    pred_name = 'nih_preds.npy'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_path = predictions_dir / pred_name
    np.save(file=pred_path, arr=y_pred_avg)

    # Get ground truth labels.
    test_true = make_true_labels(
        cxr_true_labels_path=csv_path,
        cxr_labels=cxr_labels,
        cutlabels=False # should not use because we want al¨l labels?

    )
    cxr_results = evaluate(y_pred_avg, test_true, cxr_labels)




    # Finally time for evalutating. 

if __name__ == '__main__':
    main()