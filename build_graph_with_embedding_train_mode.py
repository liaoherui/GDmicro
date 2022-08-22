import re
import os
import trans_embedding_vector
#from . import trans_embedding_vector
import preprocess_matrix_pca
import build_graph_given_matrix_with_knn_train_mode
import transform_matrix_anno


def build_dir(indir):
    if not os.path.exists(indir):
        os.makedirs(indir)

def build(infile,insample,pre,odir,kneighbor,rfile):
   p1=odir+'/P1_embedding_vector' 
   p2=odir+'/P2_pca_res'
   p3=odir+'/P3_build_graph'
   p4=odir+'/P4_node_feature'

   build_dir(p1)
   build_dir(p2)
   build_dir(p3)
   build_dir(p4)

   p1_out=p1+'/embedding_vector.txt'
   trans_embedding_vector.trans(infile,insample,p1_out)
   j1='associate.pdf'
   j2='auc_run.txt'
   if not os.path.exists(j2):
       o=open(j2,'w+')
       o.close()
   
   preprocess_matrix_pca.run_pca(j1,j2,p1_out,insample,pre,p2)
   o2=open(odir+'/build_log.txt','w+')
   build_graph_given_matrix_with_knn_train_mode.construct_g(j1,j2,p2+'/'+pre+'_matrix_ef_pca.csv',insample,kneighbor,p3,pre,o2,rfile)
   os.system('rm '+j2)
   transform_matrix_anno.trans(p2+'/'+pre+'_matrix_ef_pca.csv',p4,pre,insample)



#build('T2D_result/Graph_File/merge_embedding_Fold1.txt','../New_datasets/T2D_data_2012_Trans/T2D_meta.tsv','eggNOG','T2D_result/Graph_File/Fold1')
    
#build('../feature_eggNOG_AUS_new_embedding.txt','sample_AUS_new.txt','eggNOG','new_embedding_AUS_eggNOG')
#build('/home/heruiliao2/Deep_Learning_Project/New_methods_explore_20220403/MLP/Merge_Vector_Sp/merge_GER_sp_embedding.txt','sample_Denmark_new.txt','sp','Embedding_graph_sp_LOSO/new_embedding_GER_sp')
#build('../feature_out_test.txt','../../Graph_with_raw_data_from_paper_Merge_V2/EMG_LOO_test_China_128/sample_phenotype.txt','species','test_embedding')

    

