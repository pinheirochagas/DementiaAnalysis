from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC  # Example estimator
from sklearn.ensemble import RandomForestClassifier  # Another example estimator
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance


# def classify_and_evaluate(df, grouping_variable, classes, features, estimator='svm', scoring='accuracy', cross_validation='stratified_kfold', n_folds=5, feature_importance=False):
#     """
#     Classifies clinical diagnoses based on scores, optionally calculates feature importance, and returns objects for confusion matrix, ROC curve, etc.

#     Parameters:
#     df (pd.DataFrame): The dataframe containing the data.
#     grouping_variable (str): The name of the categorical variable for classification.
#     classes (list): The classes within the grouping variable to include in the classification.
#     features (list): The list of features for classification.
#     estimator (str): The classifier to use ('svm', 'random_forest', etc.).
#     scoring (str): The scoring method ('roc_auc', 'accuracy', etc.).
#     cross_validation (str): The cross-validation strategy ('stratified_kfold').
#     n_folds (int): The number of folds for cross-validation.
#     feature_importance (bool): Whether to calculate and return feature importance scores (default False).
    
#     Returns:
#     dict: A dictionary containing true labels, predicted labels, probabilities, scores, and optionally feature importances.
#     """
#     # Filter the dataframe for specified classes
#     df_filtered = df[df[grouping_variable].isin(classes)].reset_index(drop=True)
    
#     # Prepare data
#     X = df_filtered[features]
#     y = df_filtered[grouping_variable]
#     if len(classes) > 2 and scoring == 'roc_auc':
#         y = label_binarize(y, classes=classes)  # Binarize labels for multi-class ROC analysis
#         n_classes = y.shape[1]
#         y = y.argmax(axis=1)
    
#     print(X)
#     print(y)
#     # Define classifier
#     if estimator == 'svm':
#         classifier = SVC(probability=True)
#     elif estimator == 'random_forest':
#         classifier = RandomForestClassifier()
#     else:
#         raise ValueError("Unsupported estimator. Please add the estimator to the function.")
    
#     # Cross-validation
#     if cross_validation == 'stratified_kfold':
#         cv = StratifiedKFold(n_splits=n_folds)
#     else:
#         raise ValueError("Unsupported cross-validation strategy. Please add it to the function.")
    
#     # Perform cross-validated predictions
#     y_pred = cross_val_predict(classifier, X, y, cv=cv, method='predict')
#     y_proba = cross_val_predict(classifier, X, y, cv=cv, method='predict_proba')
#     scores = cross_val_score(classifier, X, y, cv=cv, scoring=scoring)
    
 
#     results = {
#         'true_labels': y,
#         'predicted_labels': y_pred,
#         'probabilities': y_proba,
#         'scores': scores,
#         'mean_score': np.mean(scores),
#         'std_dev_score': np.std(scores),
#         'Grouping Variable': grouping_variable,
#         'Classes': classes,
#         'Features': features,
#         'Model': estimator,
#         'Cross-Validation': cross_validation,
#         'N Folds': n_folds,
#         'Scoring': scoring
#     }

#     # Calculate feature importance if requested
#     if feature_importance:
#         feature_importances = np.zeros(len(features))

#         for train_index, test_index in cv.split(X, y):
#             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#             y_train, y_test = y[train_index], y[test_index]

#             classifier.fit(X_train, y_train)
            
#             if estimator == 'random_forest':
#                 # Direct method for models with built-in feature importance
#                 feature_importances += classifier.feature_importances_
#             elif estimator == 'svm':
#                 # Permutation importance for models without built-in feature importance
#                 r = permutation_importance(classifier, X_test, y_test, n_repeats=30)
#                 feature_importances += r.importances_mean

#         # Average feature importance over folds
#         feature_importances /= n_folds
#         feature_importance_dict = {features[i]: feature_importances[i] for i in range(len(features))}
#         results['feature_importance'] = feature_importance_dict

#     return results

def classify_and_evaluate(df, grouping_variable, classes, features, estimator='svm', scoring='roc_auc', cross_validation='stratified_kfold', n_folds=5, feature_importance=False):
    """
    Classifies clinical diagnoses based on scores, optionally calculates feature importance, and returns objects for confusion matrix, ROC curve, etc.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    grouping_variable (str): The name of the categorical variable for classification.
    classes (list): The classes within the grouping variable to include in the classification.
    features (list): The list of features for classification.
    estimator (str): The classifier to use ('svm', 'random_forest', etc.).
    scoring (str): The scoring method ('roc_auc', 'accuracy', etc.).
    cross_validation (str): The cross-validation strategy ('stratified_kfold').
    n_folds (int): The number of folds for cross-validation.
    feature_importance (bool): Whether to calculate and return feature importance scores (default False).
    
    Returns:
    dict: A dictionary containing true labels, predicted labels, probabilities, scores, and optionally feature importances.
    """
    # Filter the dataframe for specified classes
    df_filtered = df[df[grouping_variable].isin(classes)]
    
    # Prepare data
    X = df_filtered[features]
    print(X)
    y = df_filtered[grouping_variable]
    print(y)
    y = label_binarize(y, classes=classes)  # Binarize labels for multi-class ROC analysis
    n_classes = y.shape[1]
    
    # Define classifier
    if estimator == 'svm':
        classifier = SVC(probability=True)
    elif estimator == 'random_forest':
        classifier = RandomForestClassifier()
    else:
        raise ValueError("Unsupported estimator. Please add the estimator to the function.")
    
    # Cross-validation
    if cross_validation == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=n_folds)
    else:
        raise ValueError("Unsupported cross-validation strategy. Please add it to the function.")
    
    # Perform cross-validated predictions
    y_pred = cross_val_predict(classifier, X, y.argmax(axis=1), cv=cv, method='predict')
    y_proba = cross_val_predict(classifier, X, y.argmax(axis=1), cv=cv, method='predict_proba')
    scores = cross_val_score(classifier, X, y.argmax(axis=1), cv=cv, scoring=scoring)
    
    results = {
        'true_labels': y,
        'predicted_labels': y_pred,
        'probabilities': y_proba,
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_dev_score': np.std(scores),
        'Grouping Variable': grouping_variable,
        'Classes': classes,
        'Features': features,
        'Model': estimator,
        'Cross-Validation': cross_validation,
        'N Folds': n_folds,
        'Scoring': scoring
    }

    # Calculate feature importance if requested
    if feature_importance:
        feature_importances = np.zeros(len(features))

        for train_index, test_index in cv.split(X, y.argmax(axis=1)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier.fit(X_train, y_train.argmax(axis=1))
            
            if estimator == 'random_forest':
                # Direct method for models with built-in feature importance
                feature_importances += classifier.feature_importances_
            elif estimator == 'svm':
                # Permutation importance for models without built-in feature importance
                r = permutation_importance(classifier, X_test, y_test.argmax(axis=1), n_repeats=30)
                feature_importances += r.importances_mean

        # Average feature importance over folds
        feature_importances /= n_folds
        feature_importance_dict = {features[i]: feature_importances[i] for i in range(len(features))}
        results['feature_importance'] = feature_importance_dict

    return results



def classify_and_evaluate_wmaps(wmaps_path, df, grouping_variable, classes, smoothing_fwhm=.8, variance_threshold=0.1, estimator='svm', scoring='accuracy', cross_validation=True, n_folds=5, feature_importance=True, n_repeats=5, n_jobs=-1):
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



def predict_prob_single(new_wmap_path, nifti_masker, model, results):
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