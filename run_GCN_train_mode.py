import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import run_MLP_embedding_train_mode
from gcn_model_train_mode import encode_onehot,GCN,train,test,normalize,sparse_mx_to_torch_sparse_tensor
from calculate_avg_acc_of_cross_validation_train_mode import cal_acc_cv
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import random
import uuid

#exit()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cude.mannual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


def build_dir(inp):
    if not os.path.exists(inp):
        os.makedirs(inp)

def select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn):
    inmatrix=pd.read_table(eg_fs)
    inmatrix=inmatrix.iloc[:,train_idx]
    inmatrix.to_csv("tem_e.tsv",sep="\t")
    f=open("tem_e.tsv",'r')
    o=open('tem_e2.tsv','w+')
    line=f.readline().strip()
    o.write(line+'\n')
    while True:
        line=f.readline().strip()
        if not line:break
        o.write(line+'\n')
    o.close()
    os.system('rm tem_e.tsv')
    fm=open(meta,'r')
    om=open('tem_meta.tsv','w+')
    line=fm.readline()
    om.write(line)
    c=0
    while True:
        line=fm.readline().strip()
        if not line:break
        if c in train_idx:
            om.write(line+'\n')
        c+=1
    om.close()
    print('Run the command: Rscript feature_select_model_nodirect.R tem_e2.tsv tem_meta.tsv eggNOG '+disease)
    os.system('Rscript feature_select_model_nodirect.R tem_e2.tsv tem_meta.tsv eggNOG '+disease)
    os.system('mv tem_meta.tsv '+fdir+'/meta_Fold'+str(fn)+'.tsv')
    f2=open('eggNOG_feature_weight.csv','r')
    line=f2.readline()
    d={}
    t=0
    while True:
        line=f2.readline().strip()
        if not line:break
        line=re.sub('\"','',line)
        ele=re.split(',',line)
        t+=1
        if ele[-1]=='NA':continue
        if float(ele[-1])==0:continue
        d[ele[0]]=''
    print(':: Log: There are '+str(len(d))+'/'+str(t)+' features selected!\n')
    os.system('mv eggNOG_feature_weight.csv '+fdir+'/eggNOG_feature_weight_Fold'+str(fn)+'.csv')
    os.system('mv eggNOG_evaluation.pdf '+fdir+'/eggNOG_evaluation_Fold'+str(fn)+'.pdf')
    f3=open(eg_fs_norm,'r')
    line=f3.readline()
    o2=open('tem_e3.tsv','w+')
    o2.write(line)
    while True:
        line=f3.readline().strip()
        if not line:break
        ele=line.split('\t')
        if ele[0] not in d:continue
        o2.write(line+'\n')
    o2.close()
    os.system('rm tem_e2.tsv')
    os.system('mv tem_e3.tsv '+fdir+'/eggNOG_features_Fold'+str(fn)+'.tsv')
    a=fdir+'/eggNOG_features_Fold'+str(fn)+'.tsv'
    #a=fdir+'/eggNOG_features_Fold'+str(fn)+'.tsv'
    return a


#def build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn):
    
    


    

def hard_case_split(infeatures,inlabels):
    splits=StratifiedKFold(n_splits=10,shuffle=True,random_state=1234)
    dist=cosine_similarity(infeatures,infeatures)
    dist_abs=np.maximum(dist,-dist)
    did2d={} # ID -> Minimum distance
    sid=0
    for s in dist_abs:
        res=np.argsort(s)[::-1]
        for r in res:
            if r==sid:continue
            did2d[r]=s[r]
            break
        sid+=1
    res=sorted(did2d.items(), key = lambda kv:(kv[1], kv[0])) 
    res_half=res[:int(len(res)/2)]
    candidate_crc=[]
    candidate_health=[]
    #clabels=[]
    for r in res_half:
        if inlabels[r[0]]=='CRC':
            candidate_crc.append(r[0])
        else:
            candidate_health.append(r[0])
        #clabels.append(inlabels[r[0]])
    #print(candidate,clabels)
    train_val_idx=[]

    for train_idx_sk,val_idx_sk in splits.split(infeatures,inlabels):
        val_num=len(val_idx_sk)
        #train_num=len(candidate)-val_num
        crc_num=int(val_num/2)
        health_num=val_num-crc_num
        vi1=sample(candidate_crc,crc_num)
        vi2=sample(candidate_health,health_num)
        vid=vi1+vi2
        tid=[]
        for i in range(len(infeatures)):
            if i in vid:continue
            tid.append(i)
        #print(tid,vid,len(tid),len(vid))
        #print(len(train_idx_sk),len(val_idx_sk))
        #exit()
        train_val_idx.append((tid,vid))



    #print(train_val_idx[:4])
    #print(len(train_val_idx))
    #exit()
    return train_val_idx

def feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,ot,dcs,fnum):
    cround=1
    while True:
        res={}
        if cround==2:break
        for i in feature_id:
            max_val_auc=0
            i=int(i)
            if i in selected:continue
            features_tem=[[x[i]] for x in features]
            features_tem=torch.Tensor(features_tem)
            model=GCN(nfeat=features_tem.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                val_auc=train(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features_tem,adj,labels,ot,max_val_auc,rdir,fn+1,classes_dict,tid2name,0)
                val_auc==float(val_auc)
                if val_auc>max_val_auc:
                    max_val_auc=val_auc
            res[i]=float(max_val_auc)
        res2=sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        sid=1
        if cround==1:
            for r in res2:
                #if sid==fnum+1:break
                o3.write(str(sid)+'\t'+str(dcs[r[0]])+'\t'+str(r[1])+'\n')
                sid+=1
            o3.close()
        selected[res2[0][0]]=res2[0][1]
        selected_arr.append(res2[0][0])
        cround+=1
    '''
    sid=1
    for r in selected_arr:
        o4.write(str(sid)+'\t'+str(dcs[r])+'\t'+selected[r]+'\n')
        sid+=1
    o4.close()
    '''

def node_importance_check(selected,selected_arr,tem_train_id,val_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o5,o6,ot2,nnum):
    cround=1
    while True:
        res={}
        if cround==nnum+1:break
        for i in tem_train_id:
            max_val_auc=0
            i=int(i)
            if i in selected:continue
            if i in val_idx:continue
            train_idx=selected_arr+[i]
            model=GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                val_auc=train(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features,adj,labels,ot2,max_val_auc,rdir,fn+1,classes_dict,tid2name,0)
                val_auc=float(val_auc)
                if val_auc>max_val_auc:
                    max_val_auc=val_auc
            res[i]=float(max_val_auc)
        res2=sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        sid=1
        if cround==1:
            for r in res2:
                if sid==nnum+1:break
                o5.write(str(sid)+'\t'+str(tid2name[r[0]])+'\t'+str(r[1])+'\n')
                sid+=1
            o5.close()
        selected[res2[0][0]]=res2[0][1]
        selected_arr.append(res2[0][0])
        cround+=1
    sid=1
    for r in selected_arr:
        o6.write(str(sid)+'\t'+str(tid2name[r])+'\t'+str(selected[r])+'\n')
        sid+=1
    o6.close()

    


def run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,rseed,cvfold,insp,fnum,nnum,pre_features,anode):
    if not rseed==0:
        setup_seed(rseed)
    f0=open(insp,'r')
    line=f0.readline()
    dcs={}
    cs=0
    while True:
        line=f0.readline().strip()
        if not line:break
        ele=line.split('\t')
        dcs[cs]=ele[0]
        cs+=1
    idx_features_labels = np.genfromtxt("{}".format(input_fs),dtype=np.dtype(str))
    features=idx_features_labels[:, 1:-1]
    features=features.astype(float)
    features=np.array(features)

    labels_raw = idx_features_labels[:, -1]
    labels_raw=np.array(labels_raw)

    splits=StratifiedKFold(n_splits=cvfold,shuffle=True,random_state=1234)

    fdir=out+'/Feature_File'
    gdir=out+'/Graph_File'
    rdir=out+'/Res_File'
    build_dir(fdir)
    build_dir(gdir)
    build_dir(rdir)

    ofile1=rdir+'/r1.txt'
    ofile2=rdir+'/r2.txt'

    tid2name={}
    fm=open(meta,'r')
    line=fm.readline()
    tid2name={}
    test_idx=[]
    c=0
    train_id=0
    while True:
        line=fm.readline().strip()
        if not line:break
        ele=line.split()
        tid2name[c]=ele[2]
        if ele[-1]=='test':
            test_idx.append(c)
        if ele[-1]=='train' or ele[-1]=='test':
            train_id=c
        c+=1
    train_id=train_id+1
    test_idx=np.array(test_idx)
    
    o1=open(ofile1,'w+')
    fn=0
    #train_val_idx=hard_case_split(features[:train_id],labels_raw[:train_id])

    #for train_idx,val_idx in splits.split(features[:train_id],labels_raw[:train_id]):
    for train_idx,val_idx in splits.split(features[:train_id],labels_raw[:train_id]):
        #print(labels_raw[train_idx],labels_raw[val_idx])
        #exit()
        #o3=open(rdir+'/sample_prob_fold'+str(fn+1)+'.txt','w+')
        o1.write('Fold {}'.format(fn+1)+'\n')
        print('Fold {}'.format(fn+1)+', Train:'+str(len(train_idx))+' Test:'+str(len(val_idx)))
        # Select features using lasso
        
        # Select features using lasso
        if len(pre_features)==0:
            eg_fs_sf=select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn+1)
        else:
            eg_fs_sf=pre_features[fn+1]
        # Usa all features
        '''
        eg_fs_sf=eg_fs_norm
        '''
        # Train MLP on selected features 10 times and selecte the best model to build the graph
        #exit()
        #graph=run_MLP_embedding.build_graph_mlp('../New_datasets/T2D_data_2012_Trans/T2D_eggNOG_norm.txt',train_idx,val_idx,meta,disease,fn+1,gdir)
        graph=run_MLP_embedding_train_mode.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,kneighbor,rseed,rdir)
        #exit()

        # Train and testing 
        labels,classes_dict=encode_onehot(labels_raw)

        features = sp.csr_matrix(features, dtype=np.float32)
        features=torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}".format(graph),dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        feature_id=list(range(int(features.shape[1])))
        tem_train_id=list(range(train_id))
 
        model=GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
        max_val_auc=0
        #max_test_auc=0
        for epoch in range(150):
            val_auc=train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,o1,max_val_auc,rdir,fn+1,classes_dict,tid2name,1)
            if val_auc>max_val_auc:
                max_val_auc=val_auc
        #fn+=1
        ## Feature importance
        selected={}
        selected_arr=[]
        o3=open(rdir+'/feature_importance_fold'+str(fn+1)+'.txt','w+')
        #o4=open(rdir+'/feature_importance_iterative_fold'+str(fn+1)+'.txt','w+')
        uid=uuid.uuid1().hex
        ot=open(uid+'.log','w+')
        feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,ot,dcs,fnum)
        ot.close()
        os.system('rm '+uid+'.log')

        ## Node importance
        if anode==1:
            selected={}
            selected_arr=[]
            o5=open(rdir+'/node_importance_single_fold'+str(fn+1)+'.txt','w+')
            o6=open(rdir+'/node_importance_combination_fold'+str(fn+1)+'.txt','w+')
            uid=uuid.uuid1().hex
            ot2=open(uid+'.log','w+')
            node_importance_check(selected,selected_arr,tem_train_id,val_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o5,o6,ot2,nnum)
            ot2.close()
            os.system('rm '+uid+'.log')

        fn+=1

    o1.close()
    cal_acc_cv(ofile1,ofile2)


        
    

def main():
    usage="Herui's test scripts."
    parser=argparse.ArgumentParser(prog="run_GCN.py",description=usage)
    parser.add_argument('-i','--input_feature',dest='input_fs',type=str,help="The input species feature file. (Node format)")
    parser.add_argument('-e','--eggNOG_feature',dest='eg_fs',type=str,help="The input eggNOG feature file.")
    parser.add_argument('-n','--eggNOG_feature_norm',dest='eg_fs_norm',type=str,help="The input normalized eggNOG feature file.")
    parser.add_argument('-m','--metadata',dest='meta',type=str,help="The input metadata file.")
    parser.add_argument('-d','--disease',dest='disease',type=str,help="The name of the disease.")
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of test results. (Default: GCN_res")

    args=parser.parse_args()
    
    input_fs=args.input_fs
    eg_fs=args.eg_fs
    eg_fs_norm=args.eg_fs_norm
    meta=args.meta
    disease=args.disease
    out=args.outdir

    run(input_fs,eg_fs,eg_fs_norm,meta,disease,out)


'''
if __name__=="__main__":
    sys.exit(main())
'''
