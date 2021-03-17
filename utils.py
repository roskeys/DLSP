import os
import time

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from os import listdir
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class AugmentedDataset(Dataset):
    def __init__(self, *datasets):
        """
        Constructor for augmented Dataset class
        """
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        self.dataset_numbers = {
            f"{dataset.groups}_{dataset.transform}_{dataset.contrast}_{dataset.brightness}": len(dataset)
            for dataset in datasets
        }
        self.classes = {
            f"{dataset.groups}_{dataset.transform}_{dataset.contrast}_{dataset.brightness}": dataset
            for dataset in datasets
        }

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        # Generate description
        msg = "This is the augmented dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        print(msg)

    def open_img(self, group_val, transform_val, contrast_val, brightness_val, index_val):
        """
        Opens image with specified parameters.
        Parameters:
        - class_val variable should be set to 'normal', non-covid or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        Returns loaded image as a normalized Numpy array.
        """
        dataset_name = f"{group_val}_{transform_val}_{contrast_val}_{brightness_val}"
        dataset = self.classes[dataset_name]
        assert index_val < self.dataset_numbers[dataset_name]
        return dataset[index_val][0]

    def show_img(self, group_val, transform_val, contrast_val, brightness_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'non-covid' or 'covid'
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        # Open image
        im = self.open_img(group_val, transform_val, contrast_val, brightness_val, index_val)
        # Display
        plt.imshow(im.permute(1, 2, 0))

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.
        Expects an integer value index, between 0 and len(self) - 1.
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        for name, length in self.dataset_numbers.items():
            if index < length:
                return self.classes[index]
            else:
                index -= length
        raise ValueError("Index out of bound")


class Lung_Dataset(Dataset):
    def __init__(self, groups, base_dir="dataset", transform=0, contrast=1, brightness=1):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        assert groups in {'train', 'test', 'val'}, "groups must be either 'train', 'test' or 'val'"
        self.groups = groups
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        # 3 classes will be considered here (normal and infected)
        self.classes = {'normal': 0, 'non-covid': 1, 'covid': 2}

        # Path to images for different parts of the dataset
        self.dataset_paths = {
            'normal': f'./{base_dir}/{groups}/normal/',
            'non-covid': f'./{base_dir}/{groups}/infected/non-covid/',
            'covid': f'./{base_dir}/{groups}/infected/covid/'
        }
        # Number of images in each part of the dataset
        self.dataset_numbers = {
            'normal': len(listdir(self.dataset_paths['normal'])),
            'non-covid': len(listdir(self.dataset_paths['non-covid'])),
            'covid': len(listdir(self.dataset_paths['covid'])),
        }
        # Define Data Augmentation
        self.transform = int(transform)
        self.contrast = max(0, min(contrast, 2))
        self.brightness = max(0, min(brightness, 2))

    def _transform(self, im):
        """
        self.transforms is the indicator of what transform to apply.
        1 = horizontal flip
        2 = contrast change
        4 = brightness change
        """
        if self.transform <= 7 and self.transform >= 1:
            if self.transform & 1:
                im = transforms.RandomHorizontalFlip()(im)
            if self.transform & 2:
                im = transforms.ColorJitter(contrast=self.contrast)(im)
            if self.transform & 4:
                im.transform.ColorJitter(brightness=self.brightness)(im)
        return im

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        # Generate description
        msg = "This is the dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, class_val, index_val):
        """
        Opens image with specified parameters.
        Parameters:
        - class_val variable should be set to 'normal', non-covid or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        Returns loaded image as a normalized Numpy array.
        """
        err_msg = "Error - class_val variable should be set to 'normal', 'non-covid' and 'covid'"
        assert class_val in self.classes.keys(), err_msg

        max_val = self.dataset_numbers[class_val]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(self.groups, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert 0 <= index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths[class_val], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f)) / 255.
        return im

    def show_img(self, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - class_val variable should be set to 'normal', non-covid or 'covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(class_val, index_val)

        # Display
        plt.imshow(im)

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.
        Expects an integer value index, between 0 and len(self) - 1.
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        # Get item special method
        normal_range = self.dataset_numbers['normal']
        non_covid_range = self.dataset_numbers['non-covid'] + normal_range
        covid_range = self.dataset_numbers['covid'] + non_covid_range

        index = index % covid_range
        if 0 <= index < normal_range:
            class_val = 'normal'
            label = torch.Tensor([1, 0, 0])
        elif normal_range <= index < non_covid_range:
            class_val = 'non-covid'
            label = torch.Tensor([0, 1, 0])
            index = index - normal_range
        elif non_covid_range <= index < covid_range:
            class_val = 'covid'
            label = torch.Tensor([0, 0, 1])
            index = index - non_covid_range
        else:
            raise ValueError("Index larger than the max index")
        im = self.open_img(class_val, index)
        im = self._transform(Image.fromarray(im))
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label


def dataset_distribution(*datasets):
    labels = [dataset.groups for dataset in datasets]
    normal = [dataset.dataset_numbers['normal'] for dataset in datasets]
    non_covid = [dataset.dataset_numbers['non-covid'] for dataset in datasets]
    covid = [dataset.dataset_numbers['covid'] for dataset in datasets]
    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots()
    ax.bar(x - width, normal, width=width, label='normal')
    ax.bar(x, non_covid, width=width, label='non-covid')
    ax.bar(x + width, covid, width=width, label='covid')
    ax.set_ylabel("Number")
    ax.set_title("Number of data for each class")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def add_tag(nums, shift):
        for index, num in enumerate(nums):
            plt.text(index + shift, num + 0.05, num, ha='center', fontsize=11)

    add_tag(normal, -width)
    add_tag(non_covid, 0)
    add_tag(covid, width)

    fig.tight_layout()
    plt.show()


def get_normal_and_infected(y):
    """
    normal as 0, infected as 1
    :param y:
    :return:
    """
    normal = y[:, 0].unsqueeze(1)
    infected = torch.sum(y[:, 1:], dim=1, keepdim=True)
    return torch.argmax(torch.cat([normal, infected], dim=1), dim=1).unsqueeze(1).float()


def get_covid_and_non_covid(y):
    """
    non-covid as 0, covid as 1
    :param y:
    :return:
    """
    return torch.argmax(y[:, [1, 2]], dim=1).unsqueeze(1).float()


def load_model(path):
    return torch.load(path)


def plot_loss(train_loss, val_loss, accuracy, name):
    plt.figure()
    plt.title("Training loss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"plots/{name}_Loss_plot.png")
    plt.close()

    plt.figure()
    plt.title("Accuracy")
    plt.plot(accuracy)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"plots/{name}_Accuracy.png")
    plt.close()


def train_model(model, training_set, validation_set, loss_function=nn.BCELoss, categories_callback=None, epochs=1,
                optimizer_class=Adam, lr=0.001, weight_decay=0.001, batch_size=64, save_path="saved_models", cuda=True,
                print_every=1, save_every=1, logger=None):
    logger.info(f"Start training {model.name}")
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_function()
    loss_list = []
    accuracy_list = []
    for e in range(1, epochs + 1):
        total_loss = 0
        step = 0
        correct_count = 0
        total_count = 0
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = categories_callback(labels) if categories_callback else labels

            # train the model
            optimizer.zero_grad()
            prediction = model.forward(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # count the number of correct predictions
            correct_count += torch.sum(labels.detach().cpu().int() == prediction.detach().cpu().round().int())
            total_count += len(labels)
            step += 1

        average_loss = total_loss / step
        accuracy = correct_count / total_count
        loss_list.append(average_loss)
        accuracy_list.append(accuracy)
        if e % print_every == 0:
            logger.info(
                f"epoch: {e:3d} Training loss:{average_loss:.3f} Accuracy: {accuracy * 100:.1f}% Time taken: {int(time.time() - start_time)}s")
        if e % save_every == 0:
            torch.save(model, os.path.join(save_path, model.name + f"_model-{e}.h5"))


if __name__ == '__main__':
    train_set = Lung_Dataset("train")
    train_set0 = Lung_Dataset("train", transform=1)
    train_set1 = Lung_Dataset("train", transform=2)
    train_set2 = Lung_Dataset("train", transform=4)
    test_set = Lung_Dataset("test")
    val_set = Lung_Dataset("val")
    dataset_distribution(train_set, test_set, val_set)
    augmented_set = AugmentedDataset(train_set, train_set0, train_set2, train_set1)
    augmented_set.show_img(group_val="train", transform_val=4, contrast_val=1, brightness_val=1, index_val=1)
    train_set.show_img(class_val='non-covid', index_val=50)
    # from model import Resnet50
    #
    # import logging
    #
    # if not os.path.exists("logs"):
    #     os.mkdir("logs")
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)15s %(levelname)5s: %(message)s')
    #
    # stream = logging.StreamHandler()
    # stream.setLevel(logging.INFO)
    # stream.setFormatter(formatter)
    # logger.addHandler(stream)
    #
    # handler = logging.FileHandler(f'logs/runner.py_{time.strftime("%d-%H-%M-%S", time.localtime(time.time()))}.log')
    # handler.setLevel(logging.INFO)
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    #
    # train_model(Resnet50(name="normal_and_infected", hidden_dim=1024, out_dim=1), train_set, val_set,
    #             loss_function=nn.BCELoss, categories_callback=get_normal_and_infected, epochs=1, optimizer_class=Adam,
    #             lr=0.001, weight_decay=0.001, batch_size=64, save_path="saved_models", cuda=True, print_every=1,
    #             save_every=1, logger=logger)
