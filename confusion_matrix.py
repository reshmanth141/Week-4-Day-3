from typing import List, Sequence, Tuple


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], positive_label: int = 1) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute a 2x2 confusion matrix.

    Returns:
        ((TP, FP), (FN, TN))
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    tp = fp = fn = tn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == positive_label and predicted == positive_label:
            tp += 1
        elif actual == positive_label and predicted != positive_label:
            fn += 1
        elif actual != positive_label and predicted == positive_label:
            fp += 1
        else:
            tn += 1

    return (tp, fp), (fn, tn)


def explain_confusion_matrix(matrix: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
    tp, fp = matrix[0]
    fn, tn = matrix[1]

    print("Confusion Matrix:")
    print("               Predicted Positive  Predicted Negative")
    print(f"Actual Positive       {tp}                 {fn}")
    print(f"Actual Negative       {fp}                 {tn}")
    print()
    print("Definitions:")
    print(f"TP (True Positive): {tp} -> actual positive and predicted positive")
    print(f"TN (True Negative): {tn} -> actual negative and predicted negative")
    print(f"FP (False Positive): {fp} -> actual negative but predicted positive")
    print(f"FN (False Negative): {fn} -> actual positive but predicted negative")


if __name__ == "__main__":
    # Example: 10 samples, positive class is 1
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    matrix = confusion_matrix(y_true, y_pred)
    print("Actual labels:", y_true)
    print("Predicted labels:", y_pred)
    print()
    explain_confusion_matrix(matrix)
