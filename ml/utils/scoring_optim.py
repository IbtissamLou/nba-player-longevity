from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import numpy as np

def score_classifier_with_tuning(dataset, classifier, labels, param_grid, cv=5, kfold_type="kfold"):
    """
    Évalue un classificateur par validation croisée et réglage des hyperparamètres.

    :param dataset : Ensemble de données d'entités
    :param classifier : Classificateur à évaluer
    :param labels : Libellés correspondants
    :param param_grid : Grille d'hyperparamètres pour « GridSearchCV »
    :param cv : Nombre de plis pour la validation croisée (par défaut = 5)
    :param kfold_type : Type de validation croisée (« kfold » ou « stratifié »)
    :return : Aucun
    """

    # Select KFold or StratifiedKFold
    if kfold_type == "stratified":
        kf = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    else:
        kf = KFold(n_splits=cv, random_state=42, shuffle=True)

    confusion_mat = np.zeros((2, 2))

    precision_class_1, recall_class_1, f1_class_1 = 0, 0, 0
    precision_class_0, recall_class_0, f1_class_0 = 0, 0, 0

    all_predicted_labels = []
    all_true_labels = []
    best_params_list = []

    # Perform Cross-Validation
    for train_ids, test_ids in kf.split(dataset, labels):
        X_train, X_test = dataset[train_ids], dataset[test_ids]
        y_train, y_test = labels[train_ids], labels[test_ids]

        # Hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(classifier, param_grid, cv=kf, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Best model from grid search
        best_model = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)

        # Evaluate on test set
        predicted_labels = best_model.predict(X_test)

        # Update confusion matrix
        confusion_mat += confusion_matrix(y_test, predicted_labels)

        # Compute metrics for class 1
        precision_class_1 += precision_score(y_test, predicted_labels, pos_label=1)
        recall_class_1 += recall_score(y_test, predicted_labels, pos_label=1)
        f1_class_1 += f1_score(y_test, predicted_labels, pos_label=1)

        # Compute metrics for class 0
        precision_class_0 += precision_score(y_test, predicted_labels, pos_label=0)
        recall_class_0 += recall_score(y_test, predicted_labels, pos_label=0)
        f1_class_0 += f1_score(y_test, predicted_labels, pos_label=0)

        # Store predicted and true labels
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(y_test)

    # Compute average metrics
    precision_class_1 /= cv
    recall_class_1 /= cv
    f1_class_1 /= cv
    precision_class_0 /= cv
    recall_class_0 /= cv
    f1_class_0 /= cv

    # Display results
    print("\n📊 Average Confusion Matrix:")
    print(confusion_mat)

    print("\n🎯 Performance Metrics:")
    print(f"🔹 Precision (Class 1 - Long Career Players): {precision_class_1:.4f}")
    print(f"🔹 Recall (Class 1 - Majority): {recall_class_1:.4f}")
    print(f"🔹 F1-score (Class 1 - Long Career Players): {f1_class_1:.4f}")

    print(f"🔹 Precision (Class 0 - Short Career Players): {precision_class_0:.4f}")
    print(f"🔹 Recall (Class 0 - Minority): {recall_class_0:.4f}")
    print(f"🔹 F1-score (Class 0 - Short Career Players): {f1_class_0:.4f}")

    # Full classification report
    print("\n📜 Full Classification Report (average over folds):")
    print(classification_report(all_true_labels, all_predicted_labels))

