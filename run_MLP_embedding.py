import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD,Adam
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import hiddenlayer as hl
import os
import merge_embedding_vector
import build_graph_with_embedding
import random
from copy import deepcopy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True



class MLPclassifica(nn.Module):
    def __init__(self,nfeat):
        super(MLPclassifica,self).__init__()

        self.hidden1=nn.Sequential(
            nn.Linear(
                in_features=nfeat,
                #in_features=1261,
                out_features=16,
                bias=True,
            ),
            nn.ReLU()
        )

        self.hidden2=nn.Sequential(
            nn.Linear(16,10),
            nn.ReLU()
        )

        self.classifica=nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )
        self.dropout=0.5


    def forward(self,x):
        fc1=self.hidden1(x)
        self.featuremap=fc1.detach()
        fc2=self.hidden2(fc1)
        output=self.classifica(fc2)
        return fc1,fc2,output

def load_data(inmatrixf,inmetaf,train_idx,val_idx,test_idx,disease,wwl):
    train_idx=np.array(train_idx)
    val_idx=np.array(val_idx)
    test_idx=np.array(test_idx)


    inmatrix=pd.read_table(inmatrixf)
    inmatrix_train=inmatrix.iloc[:,train_idx]
    inmatrix_val=inmatrix.iloc[:,val_idx]
    inmatrix_test=inmatrix.iloc[:,test_idx]

    #print(inmatrix_train,inmatrix_train.shape)
    #exit()
    #inmatrix=inmatrix.T
    #print(inmatrix)
    #print(inmatrix.shape)
    #exit()

    inmeta=pd.read_table(inmetaf)

    labels_train=inmeta.loc[train_idx,:]["disease"]
    labels_val=inmeta.loc[val_idx,:]["disease"]
    labels_test=inmeta.loc[test_idx,:]["disease"]


    labels_train=labels_train.to_numpy()
    labels_train[labels_train==[disease]]=1
    labels_train[labels_train==['healthy']]=0
    
    labels_val=labels_val.to_numpy()
    labels_val[labels_val==[disease]]=1
    labels_val[labels_val==['healthy']]=0

    labels_test=labels_test.to_numpy()
    if wwl==1:
        labels_test[labels_test==[disease]]=1
        labels_test[labels_test==['healthy']]=0
    else:
        labels_test[labels_test==["Unknown"]]=0


    inmatrix_train=inmatrix_train.T
    inmatrix_val=inmatrix_val.T
    inmatrix_test=inmatrix_test.T

    X_train=inmatrix_train.to_numpy()
    X_val=inmatrix_val.to_numpy()
    X_test=inmatrix_test.to_numpy()
    return X_train,X_val,X_test,labels_train,labels_val,labels_test

def AUC(output,labels):
    a=output.data.numpy()
    preds=a[:,1]
    fpr,tpr,thresholds=metrics.roc_curve(np.array(labels),np.array(preds))
    auc=metrics.auc(fpr,tpr)
    return auc

def build_graph_mlp(inmatrixf,train_idx,val_idx,inmetaf,disease,fn,odir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,bsize):
    if not rseed==0:
        setup_seed(rseed)
    o=open(odir+'/train_res_stat_Fold'+str(fn)+'.txt','w+')
    ofile1=odir+'/feature_out_train_Fold'+str(fn)+'_eggNOG.txt'
    if close_cv==0:
        ofile2=odir+'/feature_out_val_Fold'+str(fn)+'_eggNOG.txt'
    ofile3=odir+'/feature_out_test_Fold'+str(fn)+'_eggNOG.txt'
    #print(train_idx,val_idx)
    #exit()
    # Load datasets
    X_train,X_val,X_test,y_train,y_val,y_test=load_data(inmatrixf,inmetaf,train_idx,val_idx,test_idx,disease,wwl)
    #print(y_train,y_test)
    #exit()
    #trans vector to tensor
    X_train_nots=torch.from_numpy(X_train.astype(np.float32))
    y_train_t=torch.from_numpy(y_train.astype(np.int64))

    if close_cv==0:
        X_val_nots=torch.from_numpy(X_val.astype(np.float32))
        y_val_t=torch.from_numpy(y_val.astype(np.int64))

    X_test_nots=torch.from_numpy(X_test.astype(np.float32))
    y_test_t=torch.from_numpy(y_test.astype(np.int64))

    train_data_nots=Data.TensorDataset(X_train_nots,y_train_t)

    train_nots_loader=Data.DataLoader(
        dataset=train_data_nots,
        batch_size=bsize,
        shuffle=True,
        num_workers=0,
    )
    max_val_acc=0
    max_val_auc=0
    max_train_acc=0
    max_train_auc=0
    max_test_acc=0
    max_test_auc=0
    for i in range(10):

        best_auc=0

        mlpc_raw=MLPclassifica(nfeat=X_train.shape[1])

        optimizer=torch.optim.Adam(mlpc_raw.parameters(),lr=0.01,weight_decay=1e-5)
        loss_func=nn.CrossEntropyLoss()

        history1=hl.History()
        canvas1=hl.Canvas()
        print_step=25


        for epoch in range(50):
            for step, (b_x, b_y) in enumerate(train_nots_loader):
                _,_,output=mlpc_raw(b_x)
                train_loss=loss_func(output,b_y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                niter=epoch*len(train_nots_loader)+step+1

                #feature_output=mlpc.featuremap.cpu()
                #feature_out=np.array(feature_output)
                _,_,output=mlpc_raw(X_val_nots)
                val_auc=AUC(output,y_val_t)
                if float(val_auc)>float(best_auc):
                    best_auc=float(val_auc)
                    mlpc=deepcopy(mlpc_raw)

                if niter%print_step==0 and close_cv==0:
                    #_,_,output=mlpc_raw(X_val_nots)
                    _,pre_lab=torch.max(output,1)
                    val_accuracy=accuracy_score(y_val_t,pre_lab)
                    val_auc=AUC(output,y_val_t)

                    '''    
                    _,_,output=mlpc(X_test_nots)
                    _,pre_lab=torch.max(output,1)
                    test_accuracy=accuracy_score(y_test_t,pre_lab)
                    test_auc=AUC(output,y_test_t)
                    #print(niter,test_accuracy)
                    #history1.log(niter,train_loss=train_loss,test_accuracy=test_accuracy,test_AUC=test_auc)
                    
                    with canvas1:
                        canvas1.draw_plot(history1["train_loss"])
                        canvas1.draw_plot(history1["test_accuracy"])
                        canvas1.draw_plot(history1["test_AUC"])
                    '''

        #plt.savefig(odir+"/result.png",dpi=400)
    
        _,_,output=mlpc(X_train_nots)
        _,pre_lab=torch.max(output,1)
        feature_output=mlpc.featuremap.cpu()
        feature_out=np.array(feature_output)
        train_acc=accuracy_score(y_train_t,pre_lab)
        train_auc=AUC(output,y_train_t)
        print("train_accuracy:",train_acc,"train_AUC:",train_auc)
        if close_cv==0:
            _,_,output=mlpc(X_val_nots)
            _,pre_lab=torch.max(output,1)
            feature_output_val=mlpc.featuremap.cpu()
            feature_out_val=np.array(feature_output_val)
            val_accuracy=accuracy_score(y_val_t,pre_lab)
            val_auc=AUC(output,y_val_t)
            print("val_accuracy:",val_accuracy,"val_AUC:",val_auc)

        _,_,output=mlpc(X_test_nots)
        _,pre_lab=torch.max(output,1)
        feature_output_test=mlpc.featuremap.cpu()
        feature_out_test=np.array(feature_output_test)
        go=0
        if wwl==1:
            test_accuracy=accuracy_score(y_test_t,pre_lab)
            test_auc=AUC(output,y_test_t)
            print("test_accuracy:",test_accuracy,"test_AUC:",test_auc)
            if close_cv==0:
                o.write("Train accuracy: "+str(train_acc)+" Train AUC: "+str(train_auc)+"\nVal accuracy: "+str(val_accuracy)+" Val AUC: "+str(val_auc)+"\nTest accuracy: "+str(test_accuracy)+" Test AUC: "+str(test_auc)+'\n')
            else:
                o.write("Train accuracy: "+str(train_acc)+" Train AUC: "+str(train_auc)+"\nTest accuracy: "+str(test_accuracy)+" Test AUC: "+str(test_auc)+'\n')
            if len(y_test_t)<13:
                if go==0:
                    max_test_acc=test_accuracy
                    max_test_auc=test_auc
                    np.savetxt(ofile1,feature_out)
                    if close_cv==0:
                        np.savetxt(ofile2,feature_out_val)
                    np.savetxt(ofile3,feature_out_test)
                    go=1
                if test_accuracy>max_test_acc:
                    max_test_acc=test_accuracy
                    max_test_auc=test_auc
                    np.savetxt(ofile1,feature_out)
                    if close_cv==0:
                        np.savetxt(ofile2,feature_out_val)
                    np.savetxt(ofile3,feature_out_test)

            else:
                if test_auc>max_test_auc:
                    max_test_acc=test_accuracy
                    max_test_auc=test_auc
                    np.savetxt(ofile1,feature_out)
                    if close_cv==0:
                        np.savetxt(ofile2,feature_out_val)
                    np.savetxt(ofile3,feature_out_test)
                if test_auc==max_test_auc and test_accuracy>max_test_acc:
                    max_test_acc=test_accuracy
                    max_test_auc=test_auc
                    np.savetxt(ofile1,feature_out)
                    if close_cv==0:
                        np.savetxt(ofile2,feature_out_val)
                    np.savetxt(ofile3,feature_out_test)
        else:
            if close_cv==0:
                o.write("Train accuracy: "+str(train_acc)+" Train AUC: "+str(train_auc)+"\nVal accuracy: "+str(val_accuracy)+" Val AUC: "+str(val_auc)+'\n')
            else:
                o.write("Train accuracy: "+str(train_acc)+" Train AUC: "+str(train_auc)+"\n")
            if close_cv==0:
                if val_auc>max_val_auc:
                    max_val_acc=val_accuracy
                    max_val_auc=val_auc
                    np.savetxt(ofile1,feature_out)
                    np.savetxt(ofile2,feature_out_val)
                    np.savetxt(ofile3,feature_out_test)
                if val_auc==max_val_auc and val_accuracy>max_val_acc:
                    max_val_acc=val_accuracy
                    max_val_auc=val_auc
                    np.savetxt(ofile1,feature_out)
                    np.savetxt(ofile2,feature_out_val)
                    np.savetxt(ofile3,feature_out_test)
            else:
                if train_auc>max_train_auc:
                    max_train_acc=train_acc
                    max_train_auc=train_auc
                    np.savetxt(ofile1,feature_out)
                    np.savetxt(ofile3,feature_out_test)
                if train_auc==max_train_auc and train_acc>max_train_acc:
                    max_train_acc=train_acc
                    max_train_auc=train_auc
                    np.savetxt(ofile1,feature_out)
                    np.savetxt(ofile3,feature_out_test)
    if close_cv==0:
        merge_embedding_vector.merge(ofile1,ofile2,ofile3,train_idx,val_idx,test_idx,odir+'/merge_embedding_Fold'+str(fn)+'.txt')
    else:
        merge_embedding_vector.merge2(ofile1,ofile3,train_idx,test_idx,odir+'/merge_embedding_Fold'+str(fn)+'.txt')
    build_graph_with_embedding.build(odir+'/merge_embedding_Fold'+str(fn)+'.txt',inmetaf,'eggNOG',odir+'/Fold'+str(fn),kneighbor,rdir+'/sample_kneighbors_all_fold'+str(fn)+'.txt')
    graph=odir+'/Fold'+str(fn)+'/P3_build_graph/eggNOG_pca_knn_graph_final.txt'
    return graph

 
    

#graph=build_graph_mlp('../New_datasets/T2D_data_2012_Trans/T2D_eggNOG_norm.txt',list(range(340)),list(range(340,363)),'../New_datasets/T2D_data_2012_Trans/T2D_meta.tsv','T2D',1,'T2D_result/Graph_File')
#print(graph)

