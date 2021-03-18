import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


def predict_2class(model, dataloader, device=None):
    """
    model: model input
    dataloader: data input
    output: predicted labels, original labels
    """
    model = model.to(device) if device is not None else model.cpu()
    result = []
    origin = []
    with torch.no_grad():
        for sample, label in dataloader:
            sample = sample.to(device) if device is not None else sample
            output = model(sample)
            predicted = output.round()
            result.append(predicted.detach().cpu())
            origin.append(label.detach().cpu())
    return torch.cat(result, dim=0), torch.cat(origin, dim=0)


def predict(model1, model2, dataloader, device=None):
    pass

def calc_accuracy(predicted_labels, truth_labels, category_callback_func):
    """
    predicted_labels: Predicted labels
    truth_labels: Ground truth
    output: accuracy calculated
    """
    truth_labels = category_callback_func(truth_labels)
    return (torch.sum(predicted_labels.int() == truth_labels.int()) / len(truth_labels)).item()

def _output_to_int(labels, mode):
    """
    predicted_labels: Predicted labels
    mode: can be sigmoid or softmax
    output: int array for truth labels, 1-d
    """
    if mode == "sigmoid":
        result = np.array(round(labels), dtype = 'int64')
        result = result.reshape(len(result))
        return result
    elif mode == "softmax":
        result = np.argmax(labels, axis = 1)
        return result     
    else:
        print("Wrong mode. Please choose from sigmoid and softmax.")

def confusion_matrix(predicted_labels, truth_labels, mode):
    """
    predicted_labels: Predicted labels
    truth_labels: Ground truth
    mode: can be sigmoid or softmax
    output: calculated confusion matrix, with an extra dimension denoting multiclass
    """
    pred = _output_to_int(predicted_labels, mode)
    truth = _output_to_int(truth_labels, mode)
    confusion_matrix = multilabel_confusion_matrix(truth, pred)
    return confusion_matrix


def eval(model, dataloader, mode, category_callback_func):
    """
    model: model input
    dataloader: data input
    output: accuracy, precision, recall, F1-score
    """
    predicted_labels, truth_labels = predict(model, dataloader)
    acc = accuracy(predicted_labels, truth_labels)
    conf_mat = confusion_matrix(predicted_labels, truth_labels, mode)[-1]
    precision = 0 if conf_mat[1, 1] == 0 else conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    recall = 0 if conf_mat[1, 1] == 0 else conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    f1score = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
    print("Evaluation of model done. Accuracy:", acc,
          "\nPrecision:", precision,
          "\nRecall:", recall,
          "F-1 score:", f1score)
    return acc, precision, recall, f1score


if __name__ == '__main__':
    from utils import Lung_Dataset
    from torch.utils.data import DataLoader
    from utils import get_normal_and_infected

    test_set = Lung_Dataset("test")
    val_set = Lung_Dataset("val")
    model = torch.load("saved_models/infected_classifier_model.h5")
    pred, target = predict(model, DataLoader(val_set, batch_size=4, shuffle=False))
    accuracy = calc_accuracy(pred, target, get_normal_and_infected)
    print(accuracy)
