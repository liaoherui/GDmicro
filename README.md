#  GDmicro - Use GCN and deep adaptation network to predict disease based on microbiome data.

### Dependencies:
* Python ==3.7.x
* R
* Required R package: SIAMCAT 
* Required python package: 



## Usage

### Use GDmicro to pre-process your data.<BR/>
 

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
  
  ```
