from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def evaluate_classifier(y_true, y_pred, average="weighted"):
    """Evaluate classifier predictions with precision, recall, and F1-score."""
    return {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
    }