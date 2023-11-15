[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/gdmicro/README.html)
#  GDmicro - Use GCN and Deep adaptation network to classify host disease status based on human gut microbiome data.

Input and core components:  __GDmicro takes microbioal compositional abundance data as input.__ It utilizes GCN and deep adaptation network to improve the classification performance and robustness. Furthermore, it can identify potential biomarkers and indicate how these biomarkers affect hosts' disease status.

You can use GDmicro to:
 1. Classify disease status for your test samples (binary classification - healthy or disease).
 2. Explore disease-related species (potential biomarkers).
 3. Explore the biomarkers' contribution to the hosts' disease status.
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
Option 1 - The first way to install GDmicro, is to use [bioconda](https://bioconda.github.io/).
Once you have bioconda environment installed, install package gdmicro:

	conda install -c bioconda gdmicro
 
 It should be noted that some commands have been replaced if you install GDmicro using bioconda. (See below)

Command (Not bioconda)    |	Command (bioconda)
------------ | ------------- 
python GDmicro.py -h | gdmicro -h


Option 2 - Also, yon can install GDmicro via [Anaconda](https://anaconda.org/) using the commands below:<BR/>

`git clone https://github.com/liaoherui/GDmicro.git`<BR/>
`cd GDmicro`<BR/>

`conda env create -f environment.yaml`<BR/>
`conda activate gdmicro`<BR/>

`python GDmicro.py -h`<BR/>


If you have installed GDmicro. Then you can run the commands below to test the program.<BR/>
`python GDmicro.py -i Input_files/CRC_10fold.csv -t 1 -d CRC -o GDmicro_Res/CRC_10fold`<BR/>
`python GDmicro.py -i Input_files/CRC_FRA_LOSO.csv -s 10 -d CRC -o GDmicro_Res/CRC_FRA_LOSO`

Note, if you meet `ResolvePackageNotFound` error when creating Conda environment, then you can try the commands below.

`conda env create -f environment_clean.yaml`<BR/>
`conda activate gdmicro`<BR/>

`python GDmicro.py -h`<BR/>


You can also reproduce all the results in the paper using the command below. (Note: all LOSO experiments are given the random seed = 10 (`-s 10`) to make sure the reproducibility of results)

`sh run_GDmicro_demo.sh`<BR/>

## Usage
### Instruction about input data.<BR/>
To use GDmicro, you are expected to provide microbioal species abundance matrix (csv format) as input. One example is given below. 

  | subject_id       | class        | disease       | study      | Acidaminococcus_intestini  | ...     |
  |--------------|--------------|--------------|------------|------------|------------|
  | S1_a_WGS | train | IBD  | GBR  |0.0  |...   |
  | S101_a_WGS | train | IBD  | GBR  |0.0  |...   |
  | S102_a_WGS | train | IBD  | GBR  |0.0  |...   |
  
The first column refers to the sample ID, the second column refers to the class of the sample (can be "train" or "test"), the third column refers to the disease label of the sample (can be 'healthy' or `<disease>` (e.g. IBD, CRC, T2D, etc)), the fourth column refers to the study name or country information of the sample, and all remaining columns refer to the species.

For missing values in the "disease" or "study" column, you can replace them with "Unknown".

For users' reference, all datasets used in the paper can be found in the "Input_files" folder.


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
    -b, --batchsize               The batch size during the training process.(default: 64)
    -e, --apply_node              If set to 1, then will apply node importance calculation, which may take a long time. (default: not use).
    -n, --node_num                How many nodes will be output during the node importance calculation process. (default:20)
    -f, --feature_num             How many features (top x features) will be analyzed during the feature contribution score calculation process. (default: x=10)
    -c, --cvfold                  The value of k in k-fold cross validation. (default: 10).
    -s, --randomseed              The random seed. (default: not use)
    -a, --domain_adapt            Whether apply domain adaptation to the test dataset. If set to 0, then will use cross-entropy loss rather than domain adaptation loss. (default: use).
    -r, --run_fi                  Whether run feature importance calculation process. If set to 0, then will not calculate the feature importance and contribution score. (default: 1)
    -o, --outdir                  Output directory of test results. (Default: GDmicro_res).
  ```
  
  ## Output
The default output folder is named "GDmicro_res", and output files in this folder are shown below.

 ```
GDmicro_res
|-Res_file/
    |-final_predict_metrics.txt
    |-sample_prob.txt
    |-sample_kneighbors_all.txt
    |-feature_importance.txt
    |-driver_sp_change.txt
```
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
This file contains the identified biomarkers' contribution scores. By default, GDmicro calculates the top 10 important features' contribution scores. Users can adjust the analyzed feature number by `-f` parameter.

  ## -Contact-
  
 If you have any questions, please email us: heruiliao2-c@my.cityu.edu.hk
 
 ## References:

how to cite this tool:
```
Liao, H., Shang, J., & Sun, Y. GDmicro: classifying host disease status with GCN and Deep adaptation network based on the human gut microbiome data. bioRxiv. 2023. https://doi.org/10.1101/2023.06.12.544696
```


