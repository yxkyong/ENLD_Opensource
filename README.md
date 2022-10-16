# ENLD_Opensource
Code of "ENLD:Efficient Noisy Label Detection in Data Lake".

![logo](https://github.com/yxkyong/ENLD_Opensource/blob/main/base_class/logo.png)
## Data Preprocess
Dataset download url: [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) [Tiny-Imagenet](https://www.kaggle.com/c/tiny-imagenet)

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

## Selection Policy & Ablation Study
