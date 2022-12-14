# ENLD_Opensource
Code of "ENLD:Efficient Noisy Label Detection in Data Lake".

<img src="https://github.com/yxkyong/ENLD_Opensource/blob/main/base_class/logo.png" width="30%" height="30%" />

**Module Steps:** Data Preprocess-> Model Generate -> Fine-grained Noisy Label Detection

## Data Preprocess
Dataset download urls: [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) [Tiny-Imagenet](https://www.kaggle.com/c/tiny-imagenet)

Divide inventory data and incremental data of each dataset and add label noise, *path:/data_preprocess/noise_generate/*:

`python noise_generate.py --dataset --data_path --save_path` 

*usage: noise_generate.py [-h] [--dataset DATASET] [--data_path DATA_PATH] [--save_path SAVE_PATH]*

Generate unbalanced incremental datasetes from incremental data, *path:/data_preprocess/divide_inremental/*:

`python split.py --dataset --data_path --save_path` 

*usage: split.py [-h] [--dataset DATASET] [--data_path DATA_PATH] [--save_path SAVE_PATH]*

## Model Generate

Init the gerneral model, path:/model_gen/:

`python generate_model.py.py --dataset --save_path --noise_rate`

*usage: generate_model.py [-h] [--dataset DATASET] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--noise_rate NOISE_RATE]*

## Fine-grained Noisy Label Detection

Evaluate and process fine-grained noisy label detection method:

`python fine_grained_noisy_label_detection.py --dataset --model_path --vote --size --iteration --noise_rate`

*usage: fine_grained_noisy_label_detection.py [-h] [--dataset DATASET] [--data_path DATA_PATH] [--model_path MODEL_PATH] [--vote VOTE] [--size SIZE] [--batch_size_set BATCH_SIZE_SET] [--iteration ITERATION] [--noise_rate NOISE_RATE]*

## Selection Strategy 

Replace the sample selection strategy in fine-grained noisy label detection, path:/enld_policy/:

`python ENLD_random.py/ENLD_entropy.py/ENLD_confidence.py/ENLD_pseudo.py`

*usage: the same as Fine-grained Noisy Label Detection*

## Ablation Study

Conduct ablation study, path:/ablation_study/: ENLD-1~5

## Missing Label Cases

path: /missing_label/

`python --ratio 0.75 ENLD_missing_label.py`

*usage: [--ratio MISSING RATE]
