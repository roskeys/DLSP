import argparse
import os
import time
import logging

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
parser.add_argument("--gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--print_every", type=int, default=1, help="print every")
parser.add_argument("--save_every", type=int, default=1, help="save every")
parser.add_argument("--C", type=float, default=0.001, help="Regularization constant")

args = parser.parse_args()
train_set = Lung_Dataset("train")
test_set = Lung_Dataset("test")
val_set = Lung_Dataset("val")

# dataset_distribution(train_set, test_set, val_set)
# augmented_set = AugmentedDataset(train_set, train_set2, train_set1)
# augmented_set.show_img(group_val="train", transform_val=2, contrast_val=1, brightness_val=1, index_val=1)
# train_set.show_img(class_val='non-covid', index_val=50)

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

model = Resnet50(name=name, hidden_dim=args.hidden_units, out_dim=2 if args.model > 1 else 3)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

try:
    train_model(model, train_loader, val_loader, loss_function=loss_func, process_y=preprocess_for_y,
                epochs=args.epochs, cuda=args.gpu, optimizer_class=Adam, lr=args.lr, weight_decay=args.C,
                save_path=args.save_dir, print_every=args.print_every, save_every=args.save_every, logger=logger)
except KeyboardInterrupt:
    pass
finally:
    torch.save(model, os.path.join(args.save_dir, model.name + "_model.h5"))
