# ENLD_Opensource
Code of "ENLD:Efficient Noisy Label Detection in Data Lake".

## Data Preprocess
Divide inventory data and incremental data of each dataset and add label noise:

path:/data_preprocess/noise_generate/

`python noise_generate.py --dataset --data_path --save_path` 

Dataset download url: [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)[Tiny-Imagenet](https://www.kaggle.com/c/tiny-imagenet)

Generate unbalanced incremental datasetes from incremental data:

path:/data_preprocess/divide_inremental/

`python split.py --dataset --data_path --save_path` 

## Model Generate

Init the gerneral model, path:/model_gen/:

`python generate_model.py.py --dataset --save_path --noise_rate`

usage: generate_model.py [-h] [--dataset DATASET] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--noise_rate NOISE_RATE]

## Fine-grained Noisy Label Detection

Evaluate and process fine-grained noisy label detection method:

`python fine_grained_noisy_label_detection.py`
