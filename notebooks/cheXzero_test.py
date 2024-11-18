
# Importing the libraries.
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval

"""
Function for testing zero shot inference with zero-shot. The small test set is downloaded.

We should start off with the example that is provided to us in the jupyter notebook. There is also another good example 
of how to do it in the code itself.

Before we do anything should we download all ot 
The evaluation is done in the following way: 
    - 
"""

# Defining the the zero shot labels and templates. By setting str after the : in  cxr_filepath: str do provide hints. A nice addtion. 
# the data will get 
cxr_filepath: str = '../data/chexpert_test.h5' # file path of chest x-ray. I need to convert 
cxr_true_labels_path: Optional[str] = '../data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path

model_dir: str = '../checkpoints/chexzero_weights' # where the pretranined weights are.
predictions_dir: Path = Path('../predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label. This is for CheXpert
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# -----------Model Paths---------------
# if we are using ensample will we collect all the models here.
model_paths = []
for subdir, dirs, files in os.walk(model_dir):   # Walk traverses the file tree from top to bottom. 
    for file in files: 
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)

print(model_paths)






#---------- Running infernce------------------
# It is running the ensembled models.
# It uses run_softmax_eval which may or may not make sense, It does for this model since each ot the classes are evaluated 
# individually with a softmax.That might make it more interesting if we want to compare models. 

def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
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

    # Compute averge predictions.
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg



"""
Create a main function for handling everything. 
This seems a bit too easy. We will see what happens.
"""

def main(): 
    

    # Call the predictions function.  We can assume that the preproccesing is done earlier. All of the inputs here come from files.
    predictions, y_pred_avg = ensemble_models(model_paths=model_paths,
                                              cxr_filepath= cxr_filepath, 
                                              cxr_labels=cxr_labels,
                                              cxr_pair_template=cxr_pair_template, 
                                              cache_dir=cache_dir)

    # saving the averaged preds. 
    pred_name = 'chexpert_preds.npy' # add name of preds
    predictions_dir = predictions_dir / pred_name
    np.save(file = predictions_dir, arr= y_pred_avg)

    """
    Doing the testing now. I probably need to look into further detail here.
    """
    # make test true 
    test_pred = y_pred_avg
    # Get the correct results essentially. Some easy datamining here. 
    test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    # Evaluating the model.
    cxr_results = evaluate(test_pred, test_true, cxr_labels)

    # Bootstrap results for 95 confidence intervals. It is not really relevant for our task.
    bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)

    # display auc with confidence interval.
    print(bootstrap_results[1])

if __name__ == '__main__':  
    main()







