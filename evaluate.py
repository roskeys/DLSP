import torch
from sklearn.metrics import multilabel_confusion_matrix


def predict(model, dataloader, convert_func=None, device=None, loss_func=None):
    """
    model: model input
    dataloader: data input
    output: predicted labels, original labels
    convert func: the process function for y
    """
    model = model.to(device) if device is not None else model.cpu()
    result, origin = [], []
    running_loss = 0
    with torch.no_grad():
        for sample, label in dataloader:
            if model.name == "covid_classifier":
                (sample, label) = convert_func(sample, label)
            else:
                label = convert_func(label)
            if label.shape[0] < 1:
                continue
            sample = sample.to(device) if device is not None else sample
            label = label.to(device) if device is not None else label
            output = model(sample)
            if loss_func:
                loss = loss_func(output, label)
                running_loss += loss.item()
            result.append(output.detach().cpu())
            origin.append(label.detach().cpu())
    result, origin = torch.cat(result, dim=0), torch.cat(origin, dim=0)
    if model.name == "infected_classifier" or model.name == "covid_classifier":
        result = result.round()
    else:
        result = torch.argmax(result, dim=1)
    if len(result.shape) > 1:
        result = result.squeeze(1)
    if len(origin.shape) > 1:
        origin = origin.squeeze(1)
    return result.int(), origin.int(), running_loss


def accuracy(pred, actual):
    """
    pred: Predicted labels
    actual: Ground truth
    """
    return (torch.sum(pred == actual) / float(len(actual))).item()


def confusion_matrix(pred, target):
    """
    pred: Predicted labels
    actual: Ground truth
    """
    return multilabel_confusion_matrix(target, pred)


def eval(model, dataloader, convert_func=None, device=None, loss_func=None):
    """
    :param model: model to evaluate
    :param dataloader: test dataloader
    :param convert_func: function for converting input for y
    :param device: running device
    :return:
    """
    pred, target, loss = predict(model, dataloader, convert_func, device, loss_func)
    acc = accuracy(pred, target)
    conf_mat = confusion_matrix(pred, target)[-1]
    precision = 0 if conf_mat[1, 1] == 0 else conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    recall = 0 if conf_mat[1, 1] == 0 else conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    f1score = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
    return acc, loss, conf_mat, precision, recall, f1score


if __name__ == '__main__':
    from utils import Lung_Dataset
    from torch.utils.data import DataLoader
    from utils import get_normal_and_infected, get_covid_and_non_covid, three_class_preprocessing

    test_set = Lung_Dataset("test")
    val_set = Lung_Dataset("val")

    model = torch.load("saved_models/covid_classifier_model.h5")
    print(eval(model, DataLoader(val_set, batch_size=4), get_covid_and_non_covid))

    model = torch.load("saved_models/infected_classifier_model.h5")
    print(eval(model, DataLoader(val_set, batch_size=4), get_normal_and_infected))

    model = torch.load("saved_models/three_class_model.h5")
    print(eval(model, DataLoader(val_set, batch_size=4), three_class_preprocessing))
