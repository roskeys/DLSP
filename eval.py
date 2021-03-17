import torch
from sklearn.metrics import multilabel_confusion_matrix

def predict(model, dataloader):
    """
    model: model input
    dataloader: data input
    output: predicted labels, original labels
    """
    result = []
    origin = []
    with torch.no_grad():
        for data in dataloader:
            sample, label = data
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            result.append(predicted)
            origin.append(label)
    return result, origin

def accuracy(predicted_labels, truth_labels):
    """
    predicted_labels: Predicted labels
    truth_labels: Ground truth
    output: accuracy calculated
    """
    with torch.no_grad():
        correct = 0
        for index in range(len(truth_labels)):
            if predicted_labels[index] == truth_labels[index]:
                correct += 1
        return (correct / len(truth_labels))

def confusion_matrix(predicted_labels, truth_labels):
    """
    predicted_labels: Predicted labels
    truth_labels: Ground truth
    output: calculated confusion matrix, with an extra dimension denoting multiclass
    """
    confusion_matrix = multilabel_confusion_matrix(truth_labels, predicted_labels)
    return confusion_matrix

def eval(model, dataloader):
    """
    model: model input
    dataloader: data input
    output: accuracy, precision, recall, F1-score
    """
    predicted_labels, truth_labels = predict(model, dataloader)
    acc = accuracy(predicted_labels, truth_labels)
    conf_mat = confusion_matrix(predicted_labels, truth_labels)[-1]
    precision = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    recall = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    f1score = 2 * precision * recall / (precision + recall)
    print("Evaluation of model done. Accuracy:", acc, 
           "\nPrecision:", precision,
           "\nRecall:", recall,
           "F-1 score:", f1score)
    return acc, precision, recall, f1score