# ISTD - 50.039 : Theory and Practice of Deep Learning small project
In this project, we created a model using pytorch to differentiate the COVID patient, non-COVID patient and normal patient. Using the X-ray images provided.

We provide two different kinds of models based on ResNet, 
a three class direct output model, or a stacking of two different binary classification models.

We implemented three different ResNet models: ResNet 50, ResNet 101, ResNet 152.

## Environment setup
The models are fully tested using pytorch==1.4.0, we suggest using the same version of pytorch to run our model.
```bash
pip install matplotlib
pip install sklearn
pip install torch==1.4.0 torchvision==0.5.0
```

## File structure
We assume you have extracted the dataset and have the following file structure:
```bash
.
├── dataset
│ ├── test
│ │ ├── infected
│ │ │ ├── covid
│ │ │ └── non-covid
│ │ └── normal
│ ├── train
│ │ ├── infected
│ │ │ ├── covid
│ │ │ └── non-covid
│ │ └── normal
│ └── val
│     ├── infected
│     │ ├── covid
│     │ └── non-covid
│     └── normal
├── dataset_demo
│ ├── test
│ │ ├── infected
│ │ └── normal
│ ├── train
│ │ ├── infected
│ │ └── normal
│ └── val
│     ├── infected
│     └── normal
├── DL small project.ipynb
├── evaluate.py
├── logs
├── model.py
├── plots
├── readme.md
├── runner.py
├── saved_models
├── Small_Project_instructions.pdf
└── utils.py
```

## Run the model
The `--classifier` option specifies the model category, which takes an integer value.
    1. three class model
    2. infacted/non-infected model
    3. covid/non-covid model

The `--resnet` option specifies the ResNet model to use, which takes an integer value, 50 for ResNet 50, 101 for ResNet 101, 152 for ResNet 152

The `--data_dir` option is to specify the data directory to load the data from

The `--epochs` option specifies the number of epochs to train the model

The `--lr` option specifies the learning rate

The `--batch_size` option specifies the batch size of the training loader, validation loader and test loader

The `--save_dir` option specifies the directory to save the trained models

The `--hidden_units` option specifies the number of hidden units before the final output layer

The `--gpu` option specifies use GPU or not

The `--print_every` option specifies the number of epochs to print the losses

The `--save_every` option specifies the gap to save the model

The `--C` option specifies the regularization term constant

The `--train` option specifies to train the models or not

The `--test` option specifies to test the model on the test set or not

The `--model_path` option specifies the path to model to test, if to test the model right after training, then this field can be left empty

The `--show_dataset_distribution` option allows user to see the training set size, validation size and test set size in a bar plot

The `--debug` option enters debug mode, in which for each epoch, the model will only train one batch of examples

The `--best_only` option will save the best model if the validation loss drops or the validation accuracy increases

## run the code
To train a ResNet 101 model from scratch, simply use:
```bash
python runner.py --classifier 2 --epochs=500 --print_every 1 --batch_size 64 --save_every 10 --gpu true --resnet 101
```

To test a saved model on the test set:
```bash
python runner.py --train false --test true --model_path saved_models\covid_classifier_101_model.h5
```

In order to reproduce all the results, please run test.sh
```bash
bash ./test.sh
```