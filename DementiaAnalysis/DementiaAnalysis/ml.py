from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC  # Example estimator
from sklearn.ensemble import RandomForestClassifier  # Another example estimator
import numpy as np

def classify_and_evaluate(df, grouping_variable, classes, features, estimator='svm', scoring='roc_auc', cross_validation='stratified_kfold', n_folds=5):
    """
    Classifies clinical diagnoses based on scores and returns objects for confusion matrix, ROC curve, etc.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    grouping_variable (str): The name of the categorical variable for classification.
    classes (list): The classes within the grouping variable to include in the classification.
    features (list): The list of features for classification.
    estimator (str): The classifier to use ('svm', 'random_forest', etc.).
    scoring (str): The scoring method ('roc_auc', 'accuracy', etc.).
    cross_validation (str): The cross-validation strategy ('stratified_kfold').
    n_folds (int): The number of folds for cross-validation.
    
    Returns:
    dict: A dictionary containing true labels, predicted labels, probabilities, and scores.
    """
    
    # Filter the dataframe for specified classes
    df_filtered = df[df[grouping_variable].isin(classes)]
    
    # Prepare data
    X = df_filtered[features]
    y = df_filtered[grouping_variable]
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
    
    # Calculate metrics for each class and compile them
    results = {'true_labels': y, 'predicted_labels': y_pred, 'probabilities': y_proba, 'scores': scores}
    

    # Compile results
    results = {
        'true_labels': y,
        'predicted_labels': y_pred,
        'probabilities': y_proba,
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_dev_score': np.std(scores),
        # Additional comprehensive summary details
        'Grouping Variable': grouping_variable,
        'Classes': classes,
        'Features': features,
        'Model': estimator,
        'Cross-Validation': cross_validation,
        'N Folds': n_folds,
        'Scoring': scoring
    }


    return results

