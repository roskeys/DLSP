#### resnet 50
### train a classifier for infected
#python runner.py --classifier 2 --epochs=300 --print_every 1 --save_every 10 --resnet 50
#
### train a classifier for covid
#python runner.py --classifier 3 --epochs=500 --print_every 1 --save_every 10 --resnet 50
#
### train a classifier for 3 different classes
#python runner.py --classifier 1 --epochs=300 --print_every 1 --save_every 10 --resnet 50


### resnet 101
## train a classifier for infected
python runner.py --classifier 2 --epochs=500 --print_every 1 --save_every 10 --resnet 101

## train a classifier for covid
python runner.py --classifier 3 --epochs=500 --print_every 1 --save_every 10 --resnet 101

## train a classifier for 3 different classes
python runner.py --classifier 1 --epochs=500 --print_every 1 --save_every 10 --resnet 101


### resnet 152
## train a classifier for infected
python runner.py --classifier 2 --epochs=500 --print_every 1 --save_every 10 --resnet 152

## train a classifier for covid
python runner.py --classifier 3 --epochs=500 --print_every 1 --save_every 10 --resnet 152

## train a classifier for 3 different classes
python runner.py --classifier 1 --epochs=500 --print_every 1 --save_every 10 --resnet 152
