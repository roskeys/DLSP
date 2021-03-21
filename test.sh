### Evaluate existing model files
### ResNet 50
## Three classes
python runner.py --train false --test true --model_path saved_models\three_class_50_model.h5

## Infected-Normal
python runner.py --train false --test true --model_path saved_models\infected_classifier_50_model.h5

## Covid-Non-Covid
python runner.py --train false --test true --model_path saved_models\covid_classifier_50_model.h5

### ResNet 101
## Three classes
python runner.py --train false --test true --model_path saved_models\three_class_101_model.h5

## Infected-Normal
python runner.py --train false --test true --model_path saved_models\infected_classifier_101_model.h5

## Covid-Non-Covid
python runner.py --train false --test true --model_path saved_models\covid_classifier_101_model.h5

### ResNet 152
## Three classes
python runner.py --train false --test true --model_path saved_models\three_class_152_model.h5

## Infected-Normal
python runner.py --train false --test true --model_path saved_models\infected_classifier_152_model.h5

## Covid-Non-Covid
python runner.py --train false --test true --model_path saved_models\covid_classifier_152_model.h5