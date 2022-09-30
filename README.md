#  GDmicro - Use GCN and deep adaptation network to predict disease based on microbiome data.

### Dependencies:
* Python ==3.7.x
* R
* Required R package: SIAMCAT 
* Required python package: 

## Install

## Usage

### Use GDmicro_preprocess to pre-process your data.<BR/>
   1. Pre-process both the training and testing data.
  `python GDmicro_preprocess.py.py -i <Input_train_dir> -b <Input_test_dir> -o <Output_dir> -d <disease>`<BR/>
   2. Pre-process training data only. (Under training mode)
  `python GDmicro_preprocess.py.py -i <Input_train_dir> -t 1 -o <Output_dir> -d <disease>`<BR/>

### Use GDmicro to predict disease for input samples.<BR/>
 

### Full command-line options

 `python GDmicro_preprocess.py.py -h`<BR/>
 ```
 GDmicro_preprocess - Normalize all input data, merge your own test data with training data, and convert combined matrices to node feature format.
 
 optional arguments:
    -h, --help                    Show help message and exit.
    -i, --input_train             The dir of input training data.
    -b, --input_test              The dir of input test data.
    -t, --train_mode              If set to 1, then will only normalize and convert all input data. This mode can only be used when input datasets are all training data. You don't need to provide the test data under this mode. (default: 0)
    -d, --disease                 The name of disease. (Note: the value should be the same as the one in your metadata file.)
    -o, --outdir                  Output directory of combined and normalized results. (Default: GDmicro_merge) 
 ```
 
  `python GDmicro.py.py -h`<BR/>
  ```
  GDmicro - Use GCN and deep adaptation network to predict disease based on microbiome data.
  
  optional arguments:
    -h, --help                    Show help message and exit.
    -i, --input_dir               The directory of all input datasets. Should be the output of GDmicro_preprocess.
    -t, --train_mode              If set to 1, then will apply k-fold cross validation to all input datasets. This mode can only be used when input datasets are all training data. The input data should be the output of the train mode of GDmicro_preprocess. (default: 0)
    -d, --disease                 The name of disease.
    -k, --kneighbor               The number of neighborhoods in the knn graph. (default: 5)
    -e, --apply_node              If set to 1, then will apply node importance calculation, which may take a long time. (default: not use).
    -c, --cvfold                  The value of k in k-fold cross validation. (default: 10).
    -s, --randomseed              The random seed. (default: not use)
    -a, --domain_adapt            Whether apply domain adaptation to the test dataset. If set to 0, then will use cross-entropy loss rather than domain adaptation loss. (default: use).
    -o, --outdir                  Output directory of test results. (Default: GDmicro_res).
  ```
  
  ## Output
