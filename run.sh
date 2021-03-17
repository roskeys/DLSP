## train a classifier for 3 different classes
python runner.py --model 1 --epochs=200 --print_every 1 --save_every 10

## train a classifier for infected
python runner.py --model 2 --epochs=200 --print_every 1 --save_every 10

## train a classifier for covid
python runner.py --model 3 --epochs=200 --print_every 1 --save_every 10