# metrics.py (from scratch)
import numpy as np

def cnfsn_mtrx(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_report(y_true, y_pred):
    cm = cnfsn_mtrx(y_true, y_pred)
    report = ""
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        support = cm[i, :].sum()
        report += f"\nClass {i}:\n"
        report += f"  Precision: {precision:.2f}\n"
        report += f"  Recall:    {recall:.2f}\n"
        report += f"  F1-Score:  {f1:.2f}\n"
        report += f"  Support:   {support}\n"

    accuracy = np.trace(cm) / np.sum(cm)
    report += f"\nOverall Accuracy: {accuracy:.2f}\n"
    return report

def roc_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tru_postv_rate_list, fls_postv_rate_list = [], []
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        tru_postv_rate = tp / (tp + fn) if (tp + fn) else 0.0
        fls_postv_rate = fp / (fp + tn) if (fp + tn) else 0.0

        tru_postv_rate_list.append(tru_postv_rate)
        fls_postv_rate_list.append(fls_postv_rate)

    return np.array(fls_postv_rate_list), np.array(tru_postv_rate_list), thresholds

def auc(fls_postv_rate, tru_postv_rate):
    return np.trapz(tru_postv_rate, fls_postv_rate)

