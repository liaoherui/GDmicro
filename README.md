#  GDmicro - Use GCN and Deep adaptation network to classify host disease status based on human gut microbiome data.

Input and core components:  __GDmicro takes microbioal compositional abundance data as input.__ It utilizes GCN and deep adaptation network to improve the classification performance and robustness. Furthermore, it can identify potential biomarkers and indicate how these biomarkers affect hosts' disease status.

You can use GDmicro to:
 1. Classify disease status for your test samples (binary classification - healthy or disease). As shown in our experiments, GDmicro has good performance even training and test data are from different studies and sampled from different countries.
 2. Explore disease-related species (potential biomarkers).
 3. Explore the biomarkers' influence on the hosts' disease status.
 4. Explore the sample relationship of your metagenomic samples through the knn graph constructed by GDmicro.

Workflow: To remove domain discrepency between training and test data, deep adaptation network will be used to learn the latent features from input compositional abundance data. Then, we will build a inter-host microbiome similarity graph based on these robust latent features, where node features are represented by species abundance features. Finally, GCN will take this graph as input and classify labels for test samples. <!---The overview of GDmicro is show below.-->


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


If you have installed GDmicro. Then you can reproduce the results in the paper using the command below. (Note: all LOSO experiment is given the random seed = 10 (`-s 10`) to make sure the reproducibility of results)

`sh run_GDmicro_demo.sh`<BR/>

## Usage
### Instruction about input data.<BR/>
To use GDmicro, you are supposed to put all your input training and test data in two folders (e.g. `<Input_train_dir>` and `<Input_test_dir>`). Here, we give an example (two folders named  `Test_datasets/Input_train`, `Test_datasets/Input_test`) about the input data format for users' reference.



  
### Use GDmicro to classify disease status for input samples.<BR/>
   1.1. Apply GDmicro to classify the disease status of your test samples.<BR/>
   
   `python GDmicro.py -i <Input_file> -d <disease> -o <Outputdir>`<BR/>
   
    
   1.2. Apply GDmicro to do the k-fold cross-validation on your training samples. (Under training mode)<BR/> 

   `python GDmicro.py -i <Input_file> -d <disease> -t 1 -o <Outputdir>`<BR/>
   
   
   ! Note, the complete demo commands using example datasets can be found in `run_GDmicro_demo.sh`
 
   

### Full command-line options
 
  `python GDmicro.py -h`<BR/>
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
    -v, --vnode                   If set to 1, then will apply domain adaptation network to node features, and use learned latent features as node features. (default: 0)
    -o, --outdir                  Output directory of test results. (Default: GDmicro_res).
  ```
  
  ## Output
The default output folder is named "GDmicro_res", and output files in this folder are shown below.

 ```
GDmicro_res
|-Graph_file/
|-Res_file/
    |-final_predict_metrics.txt
    |-sample_prob.txt
    |-sample_kneighbors_all.txt
    |-feature_importance.txt
```
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


