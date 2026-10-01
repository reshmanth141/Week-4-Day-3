def accuracy_score(y_true, y_pred):
    """Calculate the accuracy of classification predictions.

    Args:
        y_true: Sequence of true labels.
        y_pred: Sequence of predicted labels.

    Returns:
        The fraction of correctly classified instances.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true) if len(y_true) else 0.0
