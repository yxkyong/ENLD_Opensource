# ENLD_Opensource
Code of "ENLD:Efficient Noisy Label Detection in Data Lake".

## Data Preprocess
Divide inventory data and incremental data of each dataset and add label noise:

`python noise_gen.py` 

Generate unbalanced incremental datasetes from incremental data:

`python divide_incremental.py` 

## Model Generate

Init the gerneral model:

`python init_model.py`

## Fine-grained Noisy Label Detection

Evaluate and process fine-grained noisy label detection method:

`python fine_grained_noisy_label_detection.py`
