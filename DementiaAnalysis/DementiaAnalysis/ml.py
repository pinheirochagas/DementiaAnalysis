from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC  # Example estimator
from sklearn.ensemble import RandomForestClassifier  # Another example estimator
import numpy as np


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
