# Data format

### ***EggNOG Abundance Matrix File*** <a name="eggNOG_File"/>
This file mainly contains the relative abundance information of eggNOG, each row is an eggNOG, and each column is a sample.
An example of data is shown below:  
<br />
<img src="https://github.com/liaoherui/GDmicro/blob/main/Images/eggNOG.png" width = "600" height = "200" >  
<br />
The filenames must be **\<disease\>_eggNOG_matrix.csv**, where **\<disease\>** should be the prefix of tested disease that is the same as the one in metadata file. E.g. IBD, CRC, T2D, etc

### ***Species Abundance Matrix File*** <a name="sp_File"/>
This file mainly contains the relative abundance information of species, each row is a species, and each column is a sample.
An example of data is shown below:  
<br />
<img src="https://github.com/liaoherui/GDmicro/blob/main/Images/species.png" width = "600" height = "200" >  
<br />
The filenames must be **\<disease\>_sp_matrix.csv**, where **\<disease\>** should be the prefix of tested disease that is the same as the one in metadata file. E.g. IBD, CRC, T2D, etc

### ***Metadata File*** <a name="metadata_File"/>
This file mainly contains metadata information of input samples. There are five columns representing five phenotypes about input samples, "sampleID, studyName, subjectID, disease, country", where subjectID is required and needs to be consistent with the one in **\<disease\>_eggNOG_matrix.csv** and **\<disease\>_sp_matrix.csv**. For other information, you can use "Unknown" if you don't know them.

An example of data with known label is shown below:  
<br />
<img src="https://github.com/liaoherui/GDmicro/blob/main/Images/meta_with_label.png" width = "400" height = "200" >  
<br />

An example of data without known label is shown below:  
<br />
<img src="https://github.com/liaoherui/GDmicro/blob/main/Images/meta_no_label.png" width = "400" height = "200" >  
<br />

If you provide the label for the test data, then the program will automatically calculate related prediction metrics on test data for you. Otherwise, the output will only include prediction metrics on training data. The filenames must be **\<disease\>_meta.tsv**, where **\<disease\>** should be the prefix of tested disease that is the same as the one in metadata file. E.g. IBD, CRC, T2D, etc

