import torch


def calculateMetrics(train_dataloader, val_dataloader, test_dataloader, model):
    accuracy_train, precision_train, recall_train = calc(train_dataloader, model)
    accuracy_val, precision_val, recall_val = calc(val_dataloader, model)
    accuracy_test, precision_test, recall_test = calc(test_dataloader, model)

    print('Accuracy of train data: {}'.format(accuracy_train))
    print('Precision of train data: {}'.format(precision_train))
    print('Recall of train data: {}'.format(recall_train))
    print('Accuracy of val data: {}'.format(accuracy_val))
    print('Precision of val data: {}'.format(precision_val))
    print('Recall of val data: {}'.format(recall_val))
    print('Accuracy of test data: {}'.format(accuracy_test))
    print('Precision of test data: {}'.format(precision_test))
    print('Recall of test data: {}'.format(recall_test))


def calc(dataloader, model):
    accuracy = 0
    precision = 0
    recall = 0
    counter = 0
    for X, y in dataloader:
        a, p, r = calc_accuracy(model, X, y)
        accuracy += a
        precision += p
        recall += r
        counter += 1

    return accuracy / counter, precision / counter, recall / counter


# https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002
def calc_accuracy(mdl: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> (float, float, float):
    """
    Get the accuracy with respect to the most likely label

    :param mdl:
    :param X:
    :param Y:
    :return:
    """
    # get the scores for each class (or logits)
    y_logits = mdl(X)  # unnormalized probs
    # return the values & indices with the largest value in the dimension where the scores for each class is
    # get the scores with largest values & their corresponding idx (so the class that is most likely)
    max_scores, max_idx_class = mdl(X).max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    # usually 0th coordinate is batch size
    _n = X.size(0)
    assert(_n == max_idx_class.size(0))
    # calulate acc (note .item() to do float division)
    acc = (max_idx_class == Y).sum().item() / _n

    # Calculate True Positives, False Positives, False Negatives
    true_positives = ((max_idx_class == Y) & (Y == 1)).sum().item()
    true_negatives = ((max_idx_class == Y) & (Y == 0)).sum().item()
    false_positives = ((max_idx_class != Y) & (Y == 0)).sum().item()
    false_negatives = ((max_idx_class != Y) & (Y == 1)).sum().item()

    # Calculate Precision and Recall
    prec = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
    rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0

    return acc, prec, rec
