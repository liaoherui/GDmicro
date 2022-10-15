###################### Data preprocess part ############################

python GDmicro_preprocess.py -i Test_datasets/Input_train -b  Test_datasets/Input_test -o GDmicro_test_demo/Datasets/Merge_datasets  -d IBD 


python GDmicro_preprocess.py -i Test_datasets/Input_train -t 1 -o GDmicro_test_demo/Datasets/Merge_datasets_train_mode -d IBD

#######################  Prediction and 10-fold validation part ###########################


python GDmicro.py -i GDmicro_test_demo/Datasets/Merge_datasets -d IBD -o  GDmicro_test_demo/Test_prediction_result/GDmicro_res


python GDmicro.py -i GDmicro_test_demo/Datasets/Merge_datasets_train_mode -t 1 -d IBD -o GDmicro_test_demo/Test_validation_result/GDmicro_res



