#  GDmicro - Use GCN and Deep adaptation network to classify host disease status based on human gut microbiome data.

Input and core components:  __GDmicro takes eggNOG and species abundance data as input.__ It utilizes GCN and deep adaptation network to improve the classification performance and robustness.

You can use GDmicro to:
 1. Classify disease status for your test samples (binary classification - healthy or disease). As shown in our experiments, GDmicro has good performance even training and test data are from different studies and sampled from different countries.
 2. Explore disease-related species (potential biomarkers).
 3. Explore the sample relationship of your metagenomic samples through the knn graph constructed by GDmicro.

Workflow: To remove domain discrepency between training and test data, deep adaptation network will be used to learn the latent features from input eggNOG abundance data. Then, we will build a sample similarity graph based on these robust latent features, where node features are represented by species abundance features.  Finally, GCN will take this graph as input and classify labels for test samples. <!---The overview of GDmicro is show below.-->


<!---
<img src="https://github.com/liaoherui/GDmicro/blob/main/Images/GDmicro_github.png" width = "800" height = "450" >  
-->
---------------------------------------------------------------------------

### Dependencies:
* Python ==3.7.x (3.7.3 is recommended)
* R ==3.6.1


## Install
Yon can install GDmicro via [Anaconda](https://anaconda.org/) using the commands below:<BR/>

`git clone https://github.com/liaoherui/GDmicro.git`<BR/>
`cd GDmicro`<BR/>

`conda env create -f environment.yaml`<BR/>
`conda activate gdmicro`<BR/>

`python GDmicro.py -h`<BR/>
`python GDmicro_preprocess.py -h`<BR/>

If you have installed GDmicro and downloaded the [Test_datasets](https://drive.google.com/drive/u/0/folders/1Ud-cXOMBc67h1NEYtPXmkbQf8B7cQgSC). Then you can test GDmicro with the example datasets using the command below.

`sh run_GDmicro_demo.sh`<BR/>

## Usage
### Instruction about input data.<BR/>
To use GDmicro, you are supposed to put all your input training and test data in two folders (e.g. `<Input_train_dir>` and `<Input_test_dir>`). Here, we give an example (two folders named  `Test_datasets/Input_train`, `Test_datasets/Input_test`) about the input data format for users' reference.

The `Test_datasets` and all other datasets used in the paper can be downloaded through [here](https://drive.google.com/drive/u/0/folders/1Ud-cXOMBc67h1NEYtPXmkbQf8B7cQgSC).

 For both training and predicting, the data mainly consists of three files: (1)eggNOG abundance matrix file; (2)species abundance matrix file; (3)metadata file
 ```
 Test_datasets
|-Input_train/
    |-IBD_eggNOG_matrix.csv
    |-IBD_sp_matrix.csv
    |-IBD_meta.tsv
|-Input_test/
    |-IBD_eggNOG_matrix.csv
    |-IBD_sp_matrix.csv
    |-IBD_meta.tsv
 ```
 
- [Data Format Details Introduction](data_format.md)
  - [EggNOG Abundance Matrix File](data_format.md#eggNOG_File)
  - [Species Abundance Matrix File](data_format.md#sp_File)
  - [Metadata File](data_format.md#metadata_File)
 



### Use GDmicro_preprocess to pre-process your data.<BR/>
   1.1. Pre-process both the training and testing data.<BR/>
   
  `python GDmicro_preprocess.py -i <Input_train_dir> -b <Input_test_dir> -o <Output_dir> -d <disease>`<BR/>
  
   1.2. If you don't have test data, pre-process training data only. In other words, all input data should have labels. (Under training mode)<BR/>
   
  `python GDmicro_preprocess.py -i <Input_train_dir> -t 1 -o <Output_dir> -d <disease>`<BR/>
  
   ! Note, the complete demo commands using example datasets can be found in `run_GDmicro_demo.sh`
  
### Use GDmicro to classify disease status for input samples.<BR/>
   2.1. Apply GDmicro to classify the disease status of your test samples.<BR/>
   
   `python GDmicro.py -i <Input_dir> -d <disease> -o <Outputdir>`<BR/>
   
   Note: the `<Input_dir>` should be the `<Output_dir>` of 1.1.<BR/>
    
   2.2. Apply GDmicro to do the k-fold cross-validation on your training samples. (Under training mode)<BR/> 

   `python GDmicro.py -i <Input_dir> -d <disease> -t 1 -o <Outputdir>`<BR/>
   
   Note: the `<Input_dir>` should be the `<Output_dir>` of 1.2.<BR/> 
   
   ! Note, the complete demo commands using example datasets can be found in `run_GDmicro_demo.sh`
 
   

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
  GDmicro - Use GCN and deep adaptation network to classify disease status based on human gut microbiome data.
  
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
    -r, --reverse                 If set to 1, then will use functional data as node features, and compostitional data to build edges. (default: 0)
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
Feature_file
------------
This folder contains files used and files output by the feature selection program and deep adaptation network.

Graph_file
-------------------------
This folder contains files used and files output in the graph construction process.

final_predict_metrics.txt
-------------------------
This file contains the final prediction metrics including train AUC, test AUC, etc.

sample_prob.txt
---------------
This file contains the sample name, prediction probability of T/P labels, predicted labels, and true labels. Note: For test samples without true labels, the true labels are the predicted labels.

sample_kneighbors_all.txt
-------------------------
This file contains the neighbor information of each node in the constructed graph. Each line contains the sample and all its neighbors in the constructed knn graph.

feature_importance.txt
----------------------
This file contains the identified biomarkers information. All features are ranked by the importance in the descending order.

  ## -Contact-
  
 If you have any questions, please email us: heruiliao2-c@my.cityu.edu.hk


