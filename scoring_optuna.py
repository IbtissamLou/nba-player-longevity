import optuna
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

def define_search_space(trial, classifier):
    """
    Dynamically defines the hyperparameter search space for Optuna based on the classifier type.
    """
    if classifier.__class__.__name__ == "SVC":
        params = {
            "C": trial.suggest_loguniform("C", 0.01, 100),
            "kernel": trial.suggest_categorical("kernel", ["rbf"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto", 0.1, 0.01, 0.001]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
        if params["kernel"] == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        return params

    elif classifier.__class__.__name__ in ["RandomForestClassifier", "BalancedRandomForestClassifier"]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    elif classifier.__class__.__name__ == "XGBClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.3),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        }

    else:
        raise ValueError(f"Hyperparameter tuning for {classifier.__class__.__name__} is not supported.")

def objective(trial, dataset, labels, classifier, cv=5, use_stratified=False):
    """
    Objective function for Optuna to optimize hyperparameters.
    """
    params = define_search_space(trial, classifier)

    if use_stratified:
        kf = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    else:
        kf = KFold(n_splits=cv, random_state=42, shuffle=True)

    f1_scores = []

    for train_ids, test_ids in kf.split(dataset, labels):
        X_train, X_test = dataset[train_ids], dataset[test_ids]
        y_train, y_test = labels[train_ids], labels[test_ids]

        model = classifier.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, pos_label=1))

    return np.mean(f1_scores)


def score_classifier_with_optuna(dataset, classifier, labels, n_trials, cv=5, use_stratified=False):
    """
    Uses Optuna to optimize hyperparameters and evaluates classifier using cross-validation.
    
    Returns the best trained model corresponding to the printed metrics.
    """
    # Step 1: Optimize Hyperparameters using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, dataset, labels, classifier, cv, use_stratified), n_trials=n_trials)

    best_params = study.best_params
    print("\n Best Hyperparameters Found:", best_params)

    # Step 2: Initialize Cross-Validation
    kf = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True) if use_stratified else KFold(n_splits=cv, random_state=42, shuffle=True)

    confusion_mat = np.zeros((2, 2))
    precision_class_1, recall_class_1, f1_class_1 = 0, 0, 0
    precision_class_0, recall_class_0, f1_class_0 = 0, 0, 0

    all_predicted_labels = []
    all_true_labels = []

    # Step 3: Cross-Validation Evaluation
    for train_ids, test_ids in kf.split(dataset, labels):
        X_train, X_test = dataset[train_ids], dataset[test_ids]
        y_train, y_test = labels[train_ids], labels[test_ids]

        best_model = classifier.set_params(**best_params)
        best_model.fit(X_train, y_train)
        predicted_labels = best_model.predict(X_test)

        confusion_mat += confusion_matrix(y_test, predicted_labels)

        precision_class_1 += precision_score(y_test, predicted_labels, pos_label=1)
        recall_class_1 += recall_score(y_test, predicted_labels, pos_label=1)
        f1_class_1 += f1_score(y_test, predicted_labels, pos_label=1)

        precision_class_0 += precision_score(y_test, predicted_labels, pos_label=0)
        recall_class_0 += recall_score(y_test, predicted_labels, pos_label=0)
        f1_class_0 += f1_score(y_test, predicted_labels, pos_label=0)

        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(y_test)

    # Step 4: Compute Average Metrics
    precision_class_1 /= cv
    recall_class_1 /= cv
    f1_class_1 /= cv
    precision_class_0 /= cv
    recall_class_0 /= cv
    f1_class_0 /= cv

    # Step 5: Train Best Model on Full Dataset
    best_trained_model = classifier.set_params(**best_params)
    best_trained_model.fit(dataset, labels)

    print("\n📊 Average Confusion Matrix:")
    print(confusion_mat)

    print("\n🎯 Performance Metrics:")
    print(f"🔹 Precision (Class 1 - Long Career Players): {precision_class_1:.4f}")
    print(f"🔹 Recall (Class 1 - Majority): {recall_class_1:.4f}")
    print(f"🔹 F1-score (Class 1 - Long Career Players): {f1_class_1:.4f}")

    print(f"🔹 Precision (Class 0 - Short Career Players): {precision_class_0:.4f}")
    print(f"🔹 Recall (Class 0 - Minority): {recall_class_0:.4f}")
    print(f"🔹 F1-score (Class 0 - Short Career Players): {f1_class_0:.4f}")

    print("\n📜 Full Classification Report:")
    print(classification_report(all_true_labels, all_predicted_labels))

    return best_trained_model  # Return the trained model
