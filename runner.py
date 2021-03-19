import argparse
import os
import time
import logging
from evaluate import eval
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import Resnet50
from utils import Lung_Dataset, train_model, get_normal_and_infected, get_covid_and_non_covid, \
    three_class_preprocessing, dataset_distribution, AugmentedDataset

if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

if not os.path.exists("plots"):
    os.makedirs("plots")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)15s %(levelname)5s: %(message)s')

stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
stream.setFormatter(formatter)
logger.addHandler(stream)

handler = logging.FileHandler(f'logs/runner.py_{time.strftime("%d-%H-%M-%S", time.localtime(time.time()))}.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("--model", type=int, default=2,
                    help="model category:\n\t1. 3 class model\n\t2. infacted/non-infected\n\t3. covid/non-covid")
parser.add_argument("--data_dir", default="dataset", help="load data directory")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--lr", type=float, default=0.001, help="set learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="set learning rate")
parser.add_argument("--save_dir", default="saved_models", help="save model")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--gpu", type=lambda x: (str(x).lower() == 'true'), default=True, help="use gpu")
parser.add_argument("--print_every", type=int, default=1, help="print every")
parser.add_argument("--save_every", type=int, default=1, help="save every")
parser.add_argument("--C", type=float, default=0.001, help="Regularization constant")
parser.add_argument("--train", type=lambda x: (str(x).lower() == 'true'), default=True, help="Train the model")
parser.add_argument("--test", type=lambda x: (str(x).lower() == 'true'), default=True, help="Test the selected model")
parser.add_argument("--model_path", type=str, default=None, help="The model used to evaluate and test")
parser.add_argument("--test-augmentation", type=int, default=0, help="Augmentation for test set")
parser.add_argument("--show_dataset_distribution", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--predict", type=str, help="Make prediction on new image")
parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=False)
args = parser.parse_args()

if args.model == 1:
    name = "three_class"
    loss_func = nn.CrossEntropyLoss()
    preprocess_for_y = three_class_preprocessing
elif args.model == 2:
    name = "infected_classifier"
    loss_func = nn.BCELoss()
    preprocess_for_y = get_normal_and_infected
elif args.model == 3:
    name = "covid_classifier"
    loss_func = nn.BCELoss()
    preprocess_for_y = get_covid_and_non_covid
else:
    raise NotImplementedError("No such model")

train_set = Lung_Dataset("train", base_dir=args.data_dir)
test_set = Lung_Dataset("test", base_dir=args.data_dir)
val_set1 = Lung_Dataset("val", base_dir=args.data_dir)
val_set2 = Lung_Dataset("val", base_dir=args.data_dir, transform=1, contrast=1, brightness=1)
val_set3 = Lung_Dataset("val", base_dir=args.data_dir, transform=2, contrast=1, brightness=1)
val_set4 = Lung_Dataset("val", base_dir=args.data_dir, transform=4, contrast=1, brightness=1)
val_set = AugmentedDataset(val_set1, val_set2, val_set3, val_set4)

if args.train:
    model = Resnet50(name=name, hidden_dim=args.hidden_units, out_dim=2 if args.model > 1 else 3)
    if args.show_dataset_distribution:
        dataset_distribution(train_set, test_set, val_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    try:
        train_model(model, train_loader, val_loader, loss_function=loss_func, process_y=preprocess_for_y,
                    epochs=args.epochs, cuda=args.gpu, optimizer_class=Adam, lr=args.lr, weight_decay=args.C,
                    save_path=args.save_dir, print_every=args.print_every, save_every=args.save_every, logger=logger,
                    debug=args.debug)
    except KeyboardInterrupt:
        print("Waiting to exit...")
    finally:
        torch.save(model, os.path.join(args.save_dir, model.name + "_model.h5"))

if args.test:
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    if not (args.model_path and os.path.exists(args.model_path)):
        args.model_path = os.path.join(args.save_dir, os.listdir(args.save_dir)[-1])
    model = torch.load(args.model_path)
    if model.name == "three_class":
        loss_func = nn.CrossEntropyLoss()
        preprocess_for_y = three_class_preprocessing
    elif model.name == "infected_classifier":
        loss_func = nn.BCELoss()
        preprocess_for_y = get_normal_and_infected
    elif model.name == "covid_classifier":
        loss_func = nn.BCELoss()
        preprocess_for_y = get_covid_and_non_covid
    else:
        raise NotImplementedError("No such model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc, val_loss, conf_mat, precision, recall, f1score = eval(model, dataloader=test_loader,
                                                               convert_func=preprocess_for_y, device=device,
                                                               loss_func=loss_func)
    logger.info(
        f"{args.model_path} Test loss:{val_loss:.3f} Accuracy: {acc * 100:.1f}% Precision: {precision:.3f} Recall: {recall:.3f} f1score: {f1score:.3f}")

if args.predict:
    # TODO load image and make prediction, print out the category in string
    pass
