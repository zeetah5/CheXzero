

import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
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
Doing the same as for the other, but I need to modify to have a per-class hot label instead of 4 differnt values. 
Yes that need to be done, I think. I also need to think about how each type of image is in its own directory for hdf5. How will this be solved. 
"""

def create_label_file(dataset_dir, output_csv):
    # Define class names and corresponding directories.
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

    data = []
    for class_name in class_names:
        dir_path = os.path.join(dataset_dir, class_name+ '/images/')


        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                imgpath = os.path.join(dir_path, filename)
                # Create one-hot encoded labels
                label_dict = {'imgpath': imgpath}

                """
                This is the important bit since it one hot encoding the labels. I do not know 
                """
                for cname in class_names:
                    label_dict[cname] = 1 if cname == class_name else 0
                data.append(label_dict)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Label file saved to {output_csv}")
    return class_names



"""
Now, it is time for preprocessing. 
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
Time to create a hdf5 file and the label file.
I do not know if we should make 
"""
def img_and_labels_to_hdf5(csv_filepath, hdf5_filepath, resolution=320):
    df = pd.read_csv(csv_filepath)
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

    


"""
Make the true labels. It should be able to handle 
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
        # Assuming the csv has 'imgpath as the first column' if cutlabels is false.
        full_labels = full_labels.drop(columns=['imgpath'])
    y_true = full_labels.to_numpy()
    return y_true



"""
Adjusting the make function to work with the COVID dataset. 
It loads the model, then the CLIP 
"""
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
    transformations = [
        Normalize((101.48761, 101.48761, 101.48761),
                  (83.43944, 83.43944, 83.43944)),
    ]
    if pretrained:
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)

    # Create dataset
    torch_dset = COVIDDataset(
        img_path=cxr_filepath,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(torch_dset, batch_size=32, shuffle=False)

    return model, loader



# ------------ Dataset Class------------------------

"""
Handling the covid dataset.
The class reads the images and labels from the HDF5 file. 
The dataset class reads images and labels from the HDF5 file. 
The images are converted to RGB and labels are converted PyTorch tensors. 
"""
class COVIDDataset(data.Dataset):
    def __init__(
        self,
        img_path: str,
        transform=None,
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']  # What does chest x-ray indicate here
        self.transform = transform

    def __len__(self):
        return len(self.img_dset)

    # I might need to change this, but let's do this now.
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
This is an update of the function in the zero_shot file. I will need to see if this is the 
"""

def zeroshot_classifier(classnames, templates, model, context_length = 77):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            # The templates are handled here.
            texts = [template.format(classname) for template in templates]
            # picking up the clip tokenizer. I do not know if we need to do to device the way it's done. 
            texts = clip.tokenize(texts, context_length=context_length)
            # 
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim = 1)
    
    return zeroshot_weights


"""
Time to do prediction. We should use softmax here. 
I think this is correct now.
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
Time to ensemble the methods
This is where is happens. 
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
Evaluating the model. 
"""
def evaluate(y_pred, y_true, cxr_labels):
    # For multi-class, calculate metrics accordingly
    from sklearn.metrics import classification_report, roc_auc_score

    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    print(classification_report(y_true_indices, y_pred_indices, target_names=cxr_labels))

    # Compute ROC AUC for each class
    roc_auc = roc_auc_score(y_true, y_pred, average=None)
    for idx, class_name in enumerate(cxr_labels):
        print(f"ROC AUC for {class_name}: {roc_auc[idx]:.4f}")

    # If you need to return results
    results = {'classification_report': classification_report(y_true_indices, y_pred_indices, target_names=cxr_labels, output_dict=True),
               'roc_auc': roc_auc}
    return results



"""
Time for the main function.
"""
def main():


    """
    Path for the .h5 file
    """
    cxr_filepath: str = 'data/covid_images.h5'
 
    model_dir: str = './checkpoints/chexzero_weights'  # Where the pretrained weights are

    dataset_dir = '../COVID_data/COVID-19_Radiography_Dataset' # has a longer path. 
    # The csv for the output file.
    output_csv = '../chex_covid_labels.csv'
    context_length = 77 

    # might not be necessary 
    predictions_dir: Path = Path('data/predictions')  # Where to save predictions
    cache_dir: Path = predictions_dir / "cached"  # Where to cache ensembled predictions. Might not need to cache.



    # Labels and templates
    cxr_labels = ['Normal', 'COVID-19', 'Lung_opacity', 'Viral pneumonia']
    templates = ["a photo of a person with {}."]

    """
    Will need to use a separate label file for covid because this covid must have one hot encoding. 
    """
#    create_label_file(dataset_dir, output_csv)

    # Step 2: Create the HDF5 file: 


    
    #img_and_labels_to_hdf5(output_csv, cxr_filepath)

    # Step 3: Get the model paths, 
    model_paths  = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                full_dir = os.path.join(subdir, file)
                model_paths.append(full_dir)
    print("Model paths:", model_paths)

    # Step 4: Run ensemble models.
    predictions, y_pred_avg = ensemble_models(
        model_paths=model_paths,
        cxr_filepath=cxr_filepath,
        cxr_labels=cxr_labels,
        templates=templates,
        cache_dir= cache_dir
    )

    # Step 5: Save averaged predictions. 
    pred_name = 'covid_preds.npy'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_path = predictions_dir / pred_name
    np.save(file=pred_path, arr = y_pred_avg)

    # Step 6: Get ground truth labels.
    test_true = make_true_labels(
        cxr_true_labels_path=output_csv,
        cxr_labels=cxr_labels, 
        cutlabels=False  # Should not use because.
    )
    
    # Step 7: Evaluate the model.
    cxr_results = evaluate(y_pred_avg, test_true, cxr_labels)

    bootstrap_results = bootstrap(y_pred_avg, test_true, cxr_labels)

    # Display AUC with confidence intervals
    print(bootstrap_results[1]) 

    

if __name__ == '__main__':  
    main()

    
