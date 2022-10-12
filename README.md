#  GDmicro - Use GCN and deep adaptation network to predict disease based on microbiome data.

### Dependencies:
* Python ==3.7.x
* R
* Required R package: SIAMCAT 
* Required python package: 

## Install

## Usage

### Use GDmicro_preprocess to pre-process your data.<BR/>
   1.1. Pre-process both the training and testing data.<BR/>
   
  `python GDmicro_preprocess.py.py -i <Input_train_dir> -b <Input_test_dir> -o <Output_dir> -d <disease>`<BR/>
  
   1.2. If you don't have test data, pre-process training data only. In other words, all input data should have labels. (Under training mode)<BR/>
   
  `python GDmicro_preprocess.py.py -i <Input_train_dir> -t 1 -o <Output_dir> -d <disease>`<BR/>

### Use GDmicro to predict disease for input samples.<BR/>
   2.1. Apply GDmicro to predict the health status of your test samples.<BR/>
   
   `python GDmicro.py -i <Input_dir> -d <disease> -o <Outputdir>`<BR/>
   
   Note: the `<Input_dir>` should be the `<Output_dir>` of 1.1.<BR/>
    
   2.2. Apply GDmicro to do the k-fold cross-validation on your training samples. (Under training mode)<BR/> 

   `python GDmicro.py -i <Input_dir> -d <disease> -t 1 -o <Outputdir>`<BR/>
   
   Note: the `<Input_dir>` should be the `<Output_dir>` of 1.2.<BR/>

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
The default output folder is named "GDmicro_res", and output files in this folder are shown below.

 ```
GDmicro_res
|-Feature_file/
|-Graph_file/
|-Res_file/
    |-final_predict_metrics.txt
    |-sample_prob.txt
    |-sample_kneighbors_all.txt
    |-feature_importance.txt
```
final_predict_metrics.txt
-------------------------
This file contains the final prediction metrics including train AUC, test AUC, etc.

sample_prob.txt
---------------
This file contains the sample name, prediction probability of T/P labels, predicted labels, and true labels. Note: For test samples without true labels, the true labels are the predicted labels.

sample_kneighbors_all.txt
-------------------------
This file contains the neighbor information of the constructed graph. Each line contains the sample and all its neighbors in the constructed knn graph.

feature_importance.txt
----------------------
This file contains the feature importance information. All features are ranked by the importance in the descending order.


