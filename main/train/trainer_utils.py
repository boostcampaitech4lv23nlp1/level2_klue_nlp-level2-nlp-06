## TODO: 김준휘
def klue_re_micro_f1(preds, labels):
        """KLUE-RE micro f1 (except no_relation)"""
        label_list = list(self.train_dataset.label2num.keys()) # get label_list from train_dataset.
        label_indices = list(range(len(label_list)))
        if "no_relation" in label_list:
            no_relation_label_idx = label_list.index("no_relation")
            label_indices.remove(no_relation_label_idx)
        return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    num_classes = len(self.train_dataset.label2num) # get length of label_list from train-dataset.
    labels = np.eye(num_classes)[labels]

    score = np.zeros((num_classes,))
    for c in range(num_classes):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    
    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)
    
    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }