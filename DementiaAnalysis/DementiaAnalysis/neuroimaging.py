import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets
from nilearn.input_data import NiftiLabelsMasker
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.model_selection import train_test_split
from nilearn.maskers import NiftiMasker
from sklearn.feature_selection import VarianceThreshold
from nilearn.decoding import DecoderRegressor, Decoder

from nilearn.input_data import NiftiMasker


from sklearn.preprocessing import label_binarize
from nilearn.input_data import NiftiMasker
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import numpy as np
import re
from sklearn.linear_model import LogisticRegression

# # Load environment variables
# _ = load_dotenv(find_dotenv())  
# # Adjust environment variable names as needed
# API_KEY = os.environ['API_KEY_OPEN_AI']
# API_VERSION = os.environ['API_VERSION']
# RESOURCE_ENDPOINT = os.environ['RESOURCE_ENDPOINT']
# openai.api_type = "azure"
# openai.api_version = API_VERSION  # This can be overwritten with an incorrect default if not specified with some langchain objects
# openai.log = "" 


def wmap_to_atlas(nifti_img, atlas_name="dk"):
    """
    Converts the NIfTI image into an anatomical atlas.
    
    Parameters:
    - nifti_img: NIfTI image object.
    - atlas_name: Name of the atlas to use (default is "dk" for Desikan-Killiany).
    
    Returns:
    - NIfTI image object resampled to the atlas space with values averaged within each parcel.
    """
    if atlas_name == "dk":
        atlas = datasets.fetch_atlas_destrieux_2009()["maps"]
    else:
        raise ValueError("Unsupported atlas name.")
    
    resampled_img = image.resample_to_img(nifti_img, atlas)
    masker = NiftiLabelsMasker(labels_img=atlas)
    signals = masker.fit_transform(resampled_img)
    avg_img = masker.inverse_transform(signals)
    
    return avg_img

def wmap_rois(nifti_img, atlas_name="dk"):
    """
    Organizes the data into a DataFrame.
    
    Parameters:
    - nifti_img: NIfTI image object.
    - atlas_name: Name of the atlas to use (default is "dk" for Desikan-Killiany).
    
    Returns:
    - DataFrame with columns being the name of the parcels of the atlas, and values being the averaged z-score for each parcel.
    """
    if atlas_name == "dk":
        atlas = datasets.fetch_atlas_destrieux_2009()
        atlas_path = atlas["maps"]
        atlas_img = nib.load(atlas_path)  # Load the atlas image
        label_names = {label[0]: label[1] for label in atlas["labels"]}
    else:
        raise ValueError("Unsupported atlas name.")
    
    masker = NiftiLabelsMasker(labels_img=atlas_img)
    signals = masker.fit_transform(nifti_img)
    
    # Map the labels from the masker to their names
    labels_from_masker = [label_names[label] for label in masker.labels_]
    
    # Create DataFrame
    df = pd.DataFrame(signals, columns=labels_from_masker)
    
    # Convert the DataFrame to a dictionary and then to a string
    neuropsy_dic = df.to_dict(orient='dict')
    neuropsy_str = str(neuropsy_dic)

    return df, neuropsy_str

def get_wmap_filenames(dir_path):
    """
    This function retrieves the full paths of all files in a specified directory.

    Parameters:
    dir_path (str): The path to the directory from which to retrieve file paths.

    Returns:
    list: A list of strings where each string is the full path to a file in the directory specified by dir_path.
    """
    wmaps_files = []
    for path in os.listdir(dir_path):
        paths = os.path.join(dir_path, path)
        wmaps_files.append(paths)
    return wmaps_files


def classify_and_evaluate_wmap(wmaps_path, df, grouping_variable, classes, smoothing_fwhm=.8, variance_threshold=0.1, estimator='svm', scoring='accuracy', cross_validation=True, n_folds=5, feature_importance=True, n_repeats=5, n_jobs=-1):
    """
    Classifies and evaluates the data using a specified estimator and cross-validation strategy.

    Parameters:
    - wmaps_path: List of file paths to the white matter maps.
    - df: DataFrame containing the data.
    - grouping_variable: Name of the column in df that contains the grouping variable.
    - classes: List of classes to include in the classification.
    - smoothing_fwhm: Smoothing parameter for the NiftiMasker.
    - variance_threshold: Variance threshold for feature selection.
    - estimator: Estimator to use for classification (either 'svm' or 'random_forest').
    - scoring: Scoring method to use for evaluation.
    - cross_validation: Cross-validation strategy to use for evaluation.
    - n_folds: Number of folds for cross-validation.
    - feature_importance: Whether to compute feature importances.
    - n_repeats: Number of permutations for feature importance computation.

    Returns:
    - Dictionary containing the results of the classification and evaluation.
    """

    # filter data to only include the classes of interest
    df = df.loc[df[grouping_variable].isin(classes), :]
    # Create a dictionary where the keys are the numbers after 'wmap_' and the values are the file paths
    wmaps_dict = {int(re.search(r'wmap_(\d+)', wmap).group(1)): wmap for wmap in wmaps_path}
    # Create a new list of file paths in the same order as df['PIDN']
    wmaps_path = [wmaps_dict[pidn] for pidn in df['PIDN'] if pidn in wmaps_dict]

    print('Masking the data.')
    # Apply masker with smoothing and binarize the labels
    nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm, standardize=True)
    print('Masking done.')
    X = nifti_masker.fit_transform(wmaps_path)
    y = df.loc[df[grouping_variable].isin(classes), grouping_variable]
    y = label_binarize(y, classes=classes)
    if y.shape[1] == 1:  # Adjust for binary classification case
        y = np.hstack((1 - y, y))
    
    # Apply Feature Selection (Optional)
    if variance_threshold > 0:
        print('Calculating feature selection.')
        selector = VarianceThreshold(threshold=variance_threshold)
        X = selector.fit_transform(X)
        print('Feature selection complete.')
    
    # Define the classifier
    if estimator == 'svm':
        model = SVC(probability=True, kernel='linear')
    elif estimator == 'random_forest':
        model = RandomForestClassifier()
    elif estimator == 'logistic_regression':
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported estimator")
    
    # Cross-validation setup
    cv_strategy = StratifiedKFold(n_splits=n_folds) if cross_validation == 'stratified_kfold' else None

    # Evaluate the model
    print('Model predicting.')
    y_pred = cross_val_predict(model, X, y.argmax(axis=1), cv=cv_strategy, method="predict", n_jobs=n_jobs)
    if scoring == 'roc_auc' and len(classes) > 2:
        # Multi-class ROC AUC
        y_proba = cross_val_predict(model, X, y.argmax(axis=1), cv=cv_strategy, method="predict_proba", n_jobs=n_jobs)
        score = roc_auc_score(y, y_proba, multi_class="ovo", average="weighted")
    elif scoring == 'roc_auc':
        # Binary ROC AUC
        y_proba = cross_val_predict(model, X, y.argmax(axis=1), cv=cv_strategy, method="predict_proba", n_jobs=n_jobs)
        score = roc_auc_score(y[:, 1], y_proba[:, 1])
    else:
        # For other scoring methods that don't require probabilities
        score = np.mean(cross_val_score(model, X, y.argmax(axis=1), cv=cv_strategy, scoring=scoring,  n_jobs=n_jobs))
    print('Prediction complete.')
    # Fit Model
    print('Model fitting.')
    model.fit(X, y.argmax(axis=1))  # Fit model to compute feature importances
    print('Fit complete.')
    # Feature Importance (Optional)
    feature_importances = None
    if feature_importance:
        print('Calculating feature importance.')
        if estimator == 'random_forest':
            feature_importances = model.feature_importances_
        elif estimator in ['svm', 'logistic_regression']:
            # Permutation feature importance for SVM
            importance_results = permutation_importance(model, X, y.argmax(axis=1), n_repeats=n_repeats, random_state=42)
            feature_importances = importance_results.importances_mean
        print('Feature importance complete.')

    # Organize results
    results = {
        'true_labels': y,
        'predicted_labels': y_pred,
        'probabilities': y_proba if 'y_proba' in locals() else None,
        'score': score,
        'Grouping Variable': grouping_variable,
        'Classes': classes,
        'Model': estimator,
        'Cross-Validation': cross_validation,
        'N Folds': n_folds,
        'N Permutations': n_repeats if feature_importance else None,
        'Variance Threshold': variance_threshold,
        'Scoring': scoring,
        'Feature Importances': feature_importances if feature_importance else None
    }
    
    return nifti_masker, model, results



def predict_prob_single_wmap(new_wmap_path, nifti_masker, model, results):
    """
    Process a new unseen NIfTI file to predict class probability using either an SVM or Random Forest model.
    Maps predicted probabilities to provided class names for clearer interpretation.
    
    Parameters:
    - new_wmap_path (str): Path to the new, unseen NIfTI file.
    - nifti_masker (NiftiMasker): An instantiated and fitted NiftiMasker object.
    - model (trained model object): A trained model object.
    - results which contains [Classes] (list of str): List of class names corresponding to the model's classes.
    
    Returns:
    - dict: A dictionary containing 'probabilities' mapped to class names.
    """
    # Load and transform the new wmap using the NiftiMasker
    new_wmap_img = nifti_masker.transform(new_wmap_path)
    
    # Predict the class probabilities for the new wmap
    probabilities = model.predict_proba(new_wmap_img)[0]
    
    # Map probabilities to class names
    class_probabilities = dict(zip(results['Classes'], probabilities))
    
    return {'probabilities': class_probabilities}
