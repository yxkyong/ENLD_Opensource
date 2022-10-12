# ENLD_Opensource
Code of "ENLD:Efficient Noisy Label Detection in Data Lake".

## Data Preprocess
Divide inventory data and incremental data of each dataset and add label noise:

`python noise_gen` 

Generate unbalanced incremental datasetes from incremental data:

`python divide_incremental` 

## Model Generate

Init the gerneral model:

`python init_model`

## Fine-grained Noisy Label Detection

Evaluate and process fine-grained noisy label detection method:

`python fine_grained_noisy_label_detection`
