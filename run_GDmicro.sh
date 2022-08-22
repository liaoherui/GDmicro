###################### Data preprocess part ############################

#python GDmicro_preprocess.py -i Test_datasets/IBD_metaphlan2_val -b  Test_datasets/IBD_metaphlan2_test_with_label -o Test_datasets/IBD_metaphlan2_merge_wl  -d IBD &&\

#python GDmicro_preprocess.py -i Test_datasets/IBD_metaphlan2_val -b  Test_datasets/IBD_metaphlan2_test_no_label -o Test_datasets/IBD_metaphlan2_merge_nl  -d IBD &&\

#python GDmicro_preprocess.py -i Test_datasets/IBD_metaphlan2_val -t 1 -o Test_datasets/IBD_metaphlan2_val_train_mode -d IBD

#######################  Train or test part ###########################

# Train the data using 10-fold cross-validation (with/without pre-selected features) and predict on the test data with labels

#python GDmicro.py -i Test_datasets/IBD_metaphlan2_merge_wl -d IBD -o  GDmicro_test/IBD_test_wl_fs

# Train the data using 10-fold cross-validation (with/without pre-selected features) and predict on the test data without labels
#python GDmicro.py -i Test_datasets/IBD_metaphlan2_merge_nl -d IBD -o GDmicro_test/IBD_test_nl_fs

# Train the data using 10-fold cross-validation (without domain adaptation) and predict on the test data with labels
#python GDmicro.py -i Test_datasets/IBD_metaphlan2_merge_wl -d IBD -a 0 -o GDmicro_test/IBD_test_wl_MLP

# Train the data using 10-fold cross-validation (without domain adaptation) and predict on the test data without labels
#python GDmicro.py -i Test_datasets/IBD_metaphlan2_merge_nl -d IBD -a 0 -o GDmicro_test/IBD_test_nl_MLP


# Train mode - Train the data using 10-fold cross validation (without pre-selected features).
python GDmicro.py -i Test_datasets/IBD_metaphlan2_val_train_mode -t 1 -d IBD -o GDmicro_test/IBD_val_train_mode

################### CRC #######
#python GDmicro.py -i Test_datasets/CRC_motu2_merge_tem -d CRC -o  GDmicro_test/CRC_CHI_41_wl

#python GDmicro.py -i Test_datasets/CRC_motu2_merge_tem -d CRC  -o  GDmicro_test/CRC_CHI_41_wl_no_cv
