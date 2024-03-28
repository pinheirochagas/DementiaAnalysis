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


def wmap_to_atlas(nifti_img, atlas_name="destrieux"):
    """
    Converts the NIfTI image into an anatomical atlas.
    
    Parameters:
    - nifti_img: NIfTI image object.
    - atlas_name: Name of the atlas to use (default is "dk" for Desikan-Killiany).
    
    Returns:
    - NIfTI image object resampled to the atlas space with values averaged within each parcel.
    """
    if atlas_name == "destrieux":
        atlas = datasets.fetch_atlas_destrieux_2009()["maps"]
    elif atlas_name == "pauli":
        atlas = datasets.fetch_atlas_pauli_2017(version='det')["maps"]
    elif atlas_name == "seitzman":
        atlas = datasets.fetch_coords_seitzman_2018()["maps"]
    elif atlas_name == "aal":
        atlas = datasets.fetch_atlas_aal()["maps"]
    elif atlas_name == "juelich":
        atlas = datasets.fetch_atlas_juelich()["maps"]
    elif atlas_name == "harvard_oxford_subcortical":
        atlas = datasets.fetch_atlas_harvard_oxford(atlas_name='sub-maxprob-thr0-1mm')["maps"]
    elif atlas_name == "harvard_oxford_cortical":
        atlas = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr0-1mm')["maps"]

    else:
        raise ValueError("Unsupported atlas name.")
    
    resampled_img = image.resample_to_img(nifti_img, atlas)
    masker = NiftiLabelsMasker(labels_img=atlas)
    signals = masker.fit_transform(resampled_img)
    avg_img = masker.inverse_transform(signals)
    
    return avg_img

def wmap_rois(nifti_img, atlas_names=["destrieux"]):
    """
    Organizes the data into a DataFrame.
    
    Parameters:
    - nifti_img: NIfTI image object.
    - atlas_name: Name of the atlas to use (default is "dk" for Desikan-Killiany).
    
    Returns:
    - DataFrame with columns being the name of the parcels of the atlas, and values being the averaged z-score for each parcel.
    """
    dfs = []  # List to store the DataFrames for each atlas
    for atlas_name in atlas_names:
        if atlas_name == "destrieux":
            atlas = datasets.fetch_atlas_destrieux_2009()
            atlas_path = atlas["maps"]
            atlas_img = nib.load(atlas_path)  # Load the atlas image
            masker = NiftiLabelsMasker(labels_img=atlas_img)
            signals = masker.fit_transform(nifti_img)
            label_names = {label[0]: label[1] for label in atlas["labels"]}
        elif atlas_name == "aal":
            atlas = datasets.fetch_atlas_aal()
            atlas_path = atlas["maps"]
            atlas_img = nib.load(atlas_path)  # Load the atlas image
            masker = NiftiLabelsMasker(labels_img=atlas_img)
            signals = masker.fit_transform(nifti_img)
            label_names = {int(index): name for index, name in zip(masker.labels_, atlas["labels"])}
        elif atlas_name == "juelich":
            atlas = datasets.fetch_atlas_juelich(atlas_name='maxprob-thr0-2mm')
            atlas_path = atlas["filename"]
            atlas_img = nib.load(atlas_path)  # Load the atlas image
            masker = NiftiLabelsMasker(labels_img=atlas_img)
            signals = masker.fit_transform(nifti_img)
            label_names = {int(index): name for index, name in zip(masker.labels_, atlas["labels"])}
        elif atlas_name == "harvard_oxford_cortical":
            atlas = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr0-1mm')
            atlas_path = atlas["filename"]
            atlas_img = nib.load(atlas_path)  # Load the atlas image
            masker = NiftiLabelsMasker(labels_img=atlas_img)
            signals = masker.fit_transform(nifti_img)
            label_names = {int(index): name for index, name in zip(masker.labels_, atlas["labels"])}
        elif atlas_name == "harvard_oxford_subcortical":
            atlas = datasets.fetch_atlas_harvard_oxford(atlas_name='sub-maxprob-thr0-1mm')
            atlas_path = atlas["filename"]
            atlas_img = nib.load(atlas_path)  # Load the atlas image
            masker = NiftiLabelsMasker(labels_img=atlas_img)
            signals = masker.fit_transform(nifti_img)
            label_names = {int(index): name for index, name in zip(masker.labels_, atlas["labels"])}
        elif atlas_name == "pauli":
            full_names = [
                'Putamen',
                'Caudate Nucleus',
                'Nucleus Accumbens',
                'Extended Amygdala',
                'Globus Pallidus externus',
                'Globus Pallidus internus',
                'Substantia Nigra, pars compacta',
                'Red Nucleus',
                'Substantia Nigra, pars reticulata',
                'Parabrachial Pigmented Nucleus',
                'Ventral Tegmental Area',
                'Ventral Pallidum',
                'Habenular Nuclei',
                'Hypothalamus',
                'Mammillary Nucleus',
                'Subthalamic Nucleus',
            ]
            atlas = datasets.fetch_atlas_pauli_2017(version='det')
            atlas_path = atlas["maps"]
            atlas_img = nib.load(atlas_path)
            masker = NiftiLabelsMasker(labels_img=atlas_img)
            signals = masker.fit_transform(nifti_img)
            label_names = {int(index): name for index, name in enumerate(full_names, start=1)}
        else:
            raise ValueError("Unsupported atlas name.")
    
        # Map the labels from the masker to their names
        labels_from_masker = [f'{atlas_name}_atlas {label_names[label]}' for label in masker.labels_]
        # Create DataFrame
        df = pd.DataFrame(signals, columns=labels_from_masker)

        # Drop column Background or background if present
        df.drop(columns=['Background', 'background'], errors='ignore', inplace=True)

        df = df.T
        df = df.reset_index()
        df.columns = ['brain_region', 'atrophy_zscore']  # Set column names

        dfs.append(df)  # Add the DataFrame to the list

    # Concatenate all the DataFrames
    df_concat = pd.concat(dfs, axis=0)
    df_concat = df_concat.sort_values(by='atrophy_zscore', ascending=True) 
    # reset index 
    df_concat = df_concat.reset_index(drop=True)
    # convert df_concat to a string
    # Set the region column as the index
    df_dic = df_concat.set_index('brain_region')
    # Convert the DataFrame to a dictionary
    df_dic = df_dic['atrophy_zscore'].to_dict()
    # Convert the dictionary to a string
    df_str = str(df_dic)
    
    # reset index of df_concat
    return df_concat, df_str


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



