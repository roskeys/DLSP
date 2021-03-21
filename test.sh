### Evaluate existing model files
### ResNet 50
## Three classes
wget https://downloads.roskey.net/models/three_class_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/three_class_model.h5

## Infected-Normal
wget https://downloads.roskey.net/models/infected_classifier_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/infected_classifier_model.h5

## Covid-Non-Covid
wget https://downloads.roskey.net/models/covid_classifier_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/covid_classifier_model.h5

### ResNet 101
## Three classes
wget https://downloads.roskey.net/models/three_class_101_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/three_class_101_model.h5

## Infected-Normal
wget https://downloads.roskey.net/models/infected_classifier_101_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/infected_classifier_101_model.h5

## Covid-Non-Covid
wget https://downloads.roskey.net/models/covid_classifier_101_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/covid_classifier_101_model.h5

### ResNet 152
## Three classes
wget https://downloads.roskey.net/models/three_class_152_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/three_class_152_model.h5

## Infected-Normal
wget https://downloads.roskey.net/models/infected_classifier_152_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/infected_classifier_152_model.h5

## Covid-Non-Covid
wget https://downloads.roskey.net/models/covid_classifier_152_model.h5 -P saved_models
python runner.py --train false --test true --model_path saved_models/covid_classifier_152_model.h5
