#  GDmicro - Use GCN and Deep adaptation network to classify host disease status based on human gut microbiome data.

Input and core components:  __GDmicro takes microbioal compositional abundance data as input.__ It utilizes GCN and deep adaptation network to improve the classification performance and robustness. Furthermore, it can identify potential biomarkers and indicate how these biomarkers affect hosts' disease status.

You can use GDmicro to:
 1. Classify disease status for your test samples (binary classification - healthy or disease). As shown in our experiments, GDmicro has good performance even training and test data are from different studies and sampled from different countries.
 2. Explore disease-related species (potential biomarkers).
 3. Explore the biomarkers' influences on the hosts' disease status.
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


If you have installed GDmicro. Then you can run the commands below to test the program.
`python GDmicro.py -i Input_files/CRC_10fold.csv -t 1 -d CRC -o GDmicro_Res/CRC_10fold`
`python GDmicro.py -i Input_files/CRC_FRA_LOSO.csv  -s 10 -d CRC -o GDmicro_Res/CRC_FRA_LOSO`

You can also reproduce all the results in the paper using the command below. (Note: all LOSO experiment is given the random seed = 10 (`-s 10`) to make sure the reproducibility of results)

`sh run_GDmicro_demo.sh`<BR/>

## Usage
### Instruction about input data.<BR/>
To use GDmicro, you are expected to provide microbioal species abundance matrix (csv format) as input. One example is given below. All datasets used in the paper can be found in the "Input_files" folder.

  | subject_id       | class        | disease       | study      | Acidaminococcus_intestini  | ...     |
  |--------------|--------------|--------------|------------|------------|------------|
  | S1_a_WGS | train | IBD  | GBR  |0.0  |...   |
  | S101_a_WGS | train | IBD  | GBR  |0.0  |...   |
  | S102_a_WGS | train | IBD  | GBR  |0.0  |...   |
  
The first column refers to the sample ID, the second columns refers to the class of the sample (can be "train" or "test"), the third column refers to the disease label of the sample (can be 'healthy' or `<disease>` (e.g. IBD, CRC, T2D, etc)), the fourth column refers to the study name or country information of the sample, and all remaining columns refer to the species.

For missing values in the "disease" or "study" column, you can replace them with "Unknown".


### Use GDmicro to classify disease status for input samples.<BR/>
   1.1. Apply GDmicro to classify the disease status of your test samples.<BR/>
   
   `python GDmicro.py -i <Input_file> -d <disease> -o <Outputdir>`<BR/>
   
   (Note: `<disease>` refers to the name of disease in the column "disease" of input matrix. For example, `-d CRC`, `-d IBD`, `-d T2D`, etc. )
    
   1.2. Apply GDmicro to do the k-fold cross-validation on your training samples. (Under training mode)<BR/> 

   `python GDmicro.py -i <Input_file> -d <disease> -t 1 -o <Outputdir>`<BR/>
   
   
   ! Note, the complete demo commands using example datasets can be found in `run_GDmicro_demo.sh`
 
   

### Full command-line options
 
  `python GDmicro.py -h`<BR/>
  ```
  GDmicro - Use GCN and deep adaptation network to classify disease status based on human gut microbiome data.
  
  optional arguments:
    -h, --help                    Show help message and exit.
    -i, --input_dir               The directory of input csv file.
    -t, --train_mode              If set to 1, then will apply k-fold cross validation to all input datasets. This mode can only be used when input datasets all have labels and set as "train" in input file. (default: 0)
    -d, --disease                 The name of disease.
    -k, --kneighbor               The number of neighborhoods in the knn graph. (default: 5)
    -e, --apply_node              If set to 1, then will apply node importance calculation, which may take a long time. (default: not use).
    -n, --node_num                How many nodes will be output during the node importance calculation process. (default:20)
    -f, --feature_num             How many features (top x features) will be analyzed during the feature influence score calculation process. (default: x=10)
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
|-Graph_file/
|-Res_file/
    |-final_predict_metrics.txt
    |-sample_prob.txt
    |-sample_kneighbors_all.txt
    |-feature_importance.txt
    |-driver_sp_change.txt
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

driver_sp_change.txt
----------------------
This file contains the identified biomarkers' influence scores. By default, GDmicro calculates the top 10 important features' influence scores. Users can adjust the analyzed feature number by `-f` parameter.

  ## -Contact-
  
 If you have any questions, please email us: heruiliao2-c@my.cityu.edu.hk


