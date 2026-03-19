from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

def score_classifier(dataset, classifier, labels, kfold_type="stratified"):
    """
    Effectue une validation croisée en 5 étapes (K-Fold ou K-Fold stratifié) pour évaluer les performances du classificateur.

    :param dataset : Ensemble de données à traiter
    :param classifier : Classificateur à utiliser
    :param labels : Libellés utilisés pour l'apprentissage et la validation
    :param kfold_type : Type de validation croisée (« kfold » pour K-Fold, « stratified » pour K-Fold stratifié)
    :return : Aucun

    """

    # Choose the appropriate cross-validation strategy
    if kfold_type.lower() == "kfold":
        kf = KFold(n_splits=5, random_state=50, shuffle=True)
    elif kfold_type.lower() == "stratified":
        kf = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
    else:
        raise ValueError("Invalid kfold_type. Choose 'kfold' or 'stratified'.")

    confusion_mat = np.zeros((2, 2))  # For binary classification
    f1_class_1 = 0
    precision_class_1 = 0
    recall_class_1 = 0
    f1_class_0 = 0
    precision_class_0 = 0
    recall_class_0 = 0
    all_predicted_labels = []  # Store all predicted labels
    all_true_labels = []  # Store true labels

    for training_ids, test_ids in kf.split(dataset, labels):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]

        # Fit classifier
        classifier.fit(training_set, training_labels)
        predicted_labels = classifier.predict(test_set)

        # Update confusion matrix
        confusion_mat += confusion_matrix(test_labels, predicted_labels)

        # Calculate metrics for each class
        precision_class_1 += precision_score(test_labels, predicted_labels, pos_label=1)
        recall_class_1 += recall_score(test_labels, predicted_labels, pos_label=1)
        f1_class_1 += f1_score(test_labels, predicted_labels, pos_label=1)

        precision_class_0 += precision_score(test_labels, predicted_labels, pos_label=0)
        recall_class_0 += recall_score(test_labels, predicted_labels, pos_label=0)
        f1_class_0 += f1_score(test_labels, predicted_labels, pos_label=0)

        # Store predicted and true labels for final classification report
        all_predicted_labels.extend(predicted_labels)
        all_true_labels.extend(test_labels)

    # Average metrics over 5 folds
    num_folds = 5
    precision_class_1 /= num_folds
    recall_class_1 /= num_folds
    f1_class_1 /= num_folds
    precision_class_0 /= num_folds
    recall_class_0 /= num_folds
    f1_class_0 /= num_folds

    # Display Results
    print("\n📊 Confusion Matrix:")
    print(confusion_mat)
    
    print("\n🎯 Performance Metrics:")
    print(f"🔹 Precision: Class 1 (Long Career Players): {precision_class_1:.4f}")
    print(f"🔹 Recall: Class 1 (Long Career Players): {recall_class_1:.4f}")
    print(f"🔹 F1-score: Class 1 (Long Career Players): {f1_class_1:.4f}")

    print(f"🔹 Precision: Class 0 (Short Career Players): {precision_class_0:.4f}")
    print(f"🔹 Recall: Class 0 (Short Career Players): {recall_class_0:.4f}")
    print(f"🔹 F1-score: Class 0 (Short Career Players): {f1_class_0:.4f}")

    # Print classification report
    print("\n📌 Full Classification Report (Averaged Over 5 Folds):")
    print(classification_report(all_true_labels, all_predicted_labels))
