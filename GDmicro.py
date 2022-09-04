import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import run_GCN_train_mode
import run_MLP_embedding_da
import run_MLP_embedding
from gcn_model import encode_onehot,GCN,train,test,test_unknown,normalize,sparse_mx_to_torch_sparse_tensor
from calculate_avg_acc_of_cross_validation_test import cal_acc_cv
import calculate_avg_acc_of_cross_validation_train_mode
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import random
import uuid

#exit()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.mannual_seed_all(seed)
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

def feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,wwl,ot,dcs,fnum,close_cv):
    cround=1
    while True:
        res={}
        if cround==2:break
        for i in feature_id:
            max_test_auc=0
            max_val_auc=0
            i=int(i)
            if i in selected:continue
            features_tem=[[x[i]] for x in features]
            features_tem=torch.Tensor(features_tem)
            model=GCN(nfeat=features_tem.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                val_auc=train(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features_tem,adj,labels,ot,max_val_auc,rdir,fn+1,classes_dict,tid2name,wwl,0,close_cv)
                if wwl==1:
                    test_auc=test(model,test_idx,features_tem,adj,labels,ot,max_test_auc,rdir,fn+1,classes_dict,tid2name,0)
                    test_auc=float(test_auc)
                    if test_auc>max_test_auc:
                        max_test_auc=test_auc
                else:
                    val_auc=float(val_auc)
                    if val_auc>max_val_auc:
                        max_val_auc=val_auc
            if wwl==1:
                res[i]=float(max_test_auc)
            else:
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
        o4.write(str(sid)+'\t'+str(dcs[r])+'\t'+str(selected[r])+'\n')
        sid+=1
    o4.close()
    '''

def node_importance_check(selected,selected_arr,tem_train_id,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o5,o6,wwl,ot2,nnum,close_cv):
    cround=1
    while True:
        res={}
        if cround==nnum+1:break
        for i in tem_train_id:
            max_val_auc=0
            max_test_auc=0
            i=int(i)
            if i in selected:continue
            if i in val_idx:continue
            if i in test_idx:continue
            train_idx=selected_arr+[i]
            model=GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                val_auc=train(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features,adj,labels,ot2,max_val_auc,rdir,fn+1,classes_dict,tid2name,wwl,0,close_cv)
                if wwl==1:
                    test_auc=test(model,test_idx,features,adj,labels,ot2,max_test_auc,rdir,fn+1,classes_dict,tid2name,0)
                    test_auc=float(test_auc)
                    if test_auc>max_test_auc:
                        max_test_auc=test_auc
                else:
                    val_auc=float(val_auc)
                    if val_auc>max_val_auc:
                        max_val_auc=val_auc
            if wwl==1:
                res[i]=float(max_test_auc)
            else:
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


def pack_output_wwl(tem,rdir):
    temd=rdir+'/tem_files'
    if not os.path.exists(temd):
        os.makedirs(temd)
    os.system('mv '+rdir+'/*.* '+temd)
    f1=open(temd+'/r2.txt','r')
    d1={}
    d2={}
    while True:
        line=f1.readline().strip()
        if not line:break
        if re.search('val',line):continue
        if re.search('Final',line):continue
        ele=line.split()
        nline=re.sub(' best','',line)
        nline=re.sub(' of Fold '+ele[6],'',nline)
        if ele[6] not in d1:
            d1[ele[6]]=nline+'\n'
        else:
            d1[ele[6]]+=nline+'\n'
        if re.search('test AUC',line):
            d2[ele[6]]=float(ele[-1])
    res=sorted(d2.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    o=open(rdir+'/final_predict_metrics.txt','w+')
    o.write(d1[res[0][0]])
    tfold=res[0][0]
    for filename in os.listdir(temd):
        if re.search('val',filename):continue
        if re.search('prob',filename):
            fd=re.split('_',filename)[2]
            if fd=="fold"+str(tfold):
                os.system('cp '+temd+'/'+filename+' '+rdir+'/sample_prob.txt')
        else:
            ele=re.split('_',filename)
            fd=ele[-1]
            fd=re.sub('\.txt','',fd)
            pre=re.sub('\.txt','',filename)
            pre=re.sub('_'+fd,'',pre)
            if fd=="fold"+str(tfold):
                os.system('cp '+temd+'/'+filename+' '+rdir+'/'+pre+'.txt')
    if tem==0:
        os.system('rm -rf '+temd)

def pack_output_nl(tem,rdir):
    temd=rdir+'/tem_files' 
    if not os.path.exists(temd):
        os.makedirs(temd)
    os.system('mv '+rdir+"/*.* "+temd)
    f1=open(temd+'/r2.txt','r')
    d1={}
    d2={}
    d3={}
    d4={}
    while True:
        line=f1.readline().strip()
        if not line:break
        ele=line.split()
        if re.search('Final',line):continue
        if re.search('val AUC',line):
            d3[ele[6]]=float(ele[-1])
            #print(line,'auc')
        if re.search('val acc',line):
            d4[ele[6]]=float(ele[-1])
            #print(line,'acc')
        if not re.search('train',line):continue
        nline=re.sub(' best','',line)
        nline=re.sub(' of Fold '+ele[6],'',nline)
        if ele[6] not in d1:
            d1[ele[6]]=nline+'\n'
        else:
            d1[ele[6]]+=nline+'\n'
        if re.search('train AUC',line):
            d2[ele[6]]=float(ele[-1])
    res=sorted(d2.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    o=open(rdir+'/final_predict_metrics.txt','w+')
    o.write(d1[res[0][0]])
    res2=sorted(d3.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    tfold1={}
    for r in res2:
        if r[1]==res2[0][1]:
            tfold1[r[0]]=d4[r[0]]
    #print(tfold1)
    #exit()
    res3=sorted(tfold1.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    tfold={}
    for r in res3:
        if r[1]==res3[0][1]:
            tfold['fold'+str(r[0])]=''
    #print(d3,d4,tfold)
    #exit()
    samples=[]
    prob={} #
    dlabel={} 
    ffold=res3[0][0]
    for filename in os.listdir(temd):
        if re.search('val',filename):continue
        if re.search('prob',filename):
            ele=re.split('_',filename)
            if ele[2] in tfold:
                f2=open(temd+'/'+filename,'r')
                while True:
                    line=f2.readline().strip()
                    if not line:break
                    ele=line.split('\t')
                    if ele[0] not in samples:
                        samples.append(ele[0])
                    if ele[0] not in prob:
                        prob[ele[0]]=[float(ele[1]),float(ele[2])]
                    else:
                        prob[ele[0]][0]+=float(ele[1])
                        prob[ele[0]][1]+=float(ele[2])
                    dlabel[int(ele[-2])]=ele[-1]

        else:
            ele=re.split('_',filename)
            fd=ele[-1]
            fd=re.sub('\.txt','',fd)
            pre=re.sub('\.txt','',filename)
            pre=re.sub('_'+fd,'',pre)
            if fd=="fold"+str(ffold):
                os.system('cp '+temd+'/'+filename+' '+rdir+'/'+pre+'.txt')
    o2=open(rdir+'/sample_prob.txt','w+')
    for s in samples:
        a=prob[s][0]/(prob[s][0]+prob[s][1])
        b=prob[s][1]/(prob[s][0]+prob[s][1])
        o2.write(s+'\t'+str(a)+'\t'+str(b))
        if prob[s][0]>prob[s][1]:
            o2.write('\t0\t'+str(dlabel[0])+'\n')
        else:
            o2.write('\t1\t'+str(dlabel[1])+'\n')
    if tem==0:
        os.system('rm -rf '+temd)



def run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,pre_features,rseed,cvfold,doadpt,insp,fnum,nnum,close_cv,anode):
    if not rseed==0:
        setup_seed(rseed)
    # Load species name -> for feature importance
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

    wwl=1 # 1 means "with label", 0 means "without label"
    if "Unknown" in labels_raw:
        wwl=0

    #print(labels_raw)
    #exit()

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
    ccv_train_idx=[]
    c=0
    train_id=0
    while True:
        line=fm.readline().strip()
        if not line:break
        ele=line.split()
        tid2name[c]=ele[2]
        if ele[-1]=='test':
            test_idx.append(c)
        if ele[-1]=='train':
            train_id=c
            ccv_train_idx.append(c)
        c+=1
    train_id=train_id+1
    test_idx=np.array(test_idx)
    #print(ccv_train_idx)
    #exit()
    
    o1=open(ofile1,'w+')
    fn=0
    #train_val_idx=hard_case_split(features[:train_id],labels_raw[:train_id])

    #for train_idx,val_idx in splits.split(features[:train_id],labels_raw[:train_id]):
    if close_cv==1:
        datasets=[(ccv_train_idx,test_idx)]

    else:
        datasets=splits.split(features[:train_id],labels_raw[:train_id])

    for train_idx,val_idx in datasets:
        #print(train_idx)
        #exit()
        #o3=open(rdir+'/sample_prob_fold'+str(fn+1)+'.txt','w+')
        o1.write('Fold {}'.format(fn+1)+'\n')
        print('Fold {}'.format(fn+1)+', Train:'+str(len(train_idx))+' Test:'+str(len(val_idx)))
        # Select features using lasso
        
        # Select features using lasso
        if len(pre_features)==0:
            eg_fs_sf=select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn+1)
        else:
            if len(pre_features)==cvfold:
                eg_fs_sf=pre_features[fn+1]
            else:
                eg_fs_sf=select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn+1)
             
        #eg_fs_sf='CRC_41_GCN/FRA_k10/Feature_File/eggNOG_features_Fold1.tsv'
        # Usa all features
        '''
        eg_fs_sf=eg_fs_norm
        '''
        # Train MLP on selected features 10 times and selecte the best model to build the graph
        #exit()
        #graph=run_MLP_embedding.build_graph_mlp('../New_datasets/T2D_data_2012_Trans/T2D_eggNOG_norm.txt',train_idx,val_idx,meta,disease,fn+1,gdir)
        if doadpt==1:
            graph=run_MLP_embedding_da.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv)
        else:
            graph=run_MLP_embedding.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv)
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
        max_test_auc=0
        for epoch in range(150):
            val_auc=train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,o1,max_val_auc,rdir,fn+1,classes_dict,tid2name,wwl,1,close_cv)
            raw_mval_auc=max_val_auc
            if val_auc>max_val_auc:
                max_val_auc=val_auc
            ### New part for testing datasets
            if wwl==1 and close_cv==0:
                test_auc=test(model,test_idx,features,adj,labels,o1,max_test_auc,rdir,fn+1,classes_dict,tid2name,1)
                if test_auc>max_test_auc:
                    max_test_auc=test_auc
            else:
                if wwl==1:
                    test_auc=test(model,test_idx,features,adj,labels,o1,max_test_auc,rdir,fn+1,classes_dict,tid2name,1)
                    if test_auc>max_test_auc:
                        max_test_auc=test_auc
                else:
                    if val_auc>raw_mval_auc:
                        test_unknown(model,test_idx,features,adj,rdir,fn+1,classes_dict,tid2name,1)
        
        ##### Feature importance
        
        selected={}
        selected_arr=[]
        o3=open(rdir+'/feature_importance_fold'+str(fn+1)+'.txt','w+')
        #o4=open(rdir+'/feature_importance_iterative_fold'+str(fn+1)+'.txt','w+')
        uid=uuid.uuid1().hex
        ot=open(uid+'.log','w+')
        feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,wwl,ot,dcs,fnum,close_cv)
        ot.close()
        os.system('rm '+uid+'.log')
        

        ##### Node importance
        if anode==1:
            selected={}
            selected_arr=[]
            o5=open(rdir+'/node_importance_single_fold'+str(fn+1)+'.txt','w+')
            o6=open(rdir+'/node_importance_combination_fold'+str(fn+1)+'.txt','w+')
            uid=uuid.uuid1().hex
            ot2=open(uid+'.log','w+')
            node_importance_check(selected,selected_arr,tem_train_id,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o5,o6,wwl,ot2,nnum,close_cv)
            ot2.close()
            os.system('rm '+uid+'.log')
        

        fn+=1

        #exit()
    o1.close()
    if wwl==1:
        if close_cv==0:
            cal_acc_cv(ofile1,ofile2)
    else:
        if close_cv==0:
            calculate_avg_acc_of_cross_validation_train_mode.cal_acc_cv(ofile1,ofile2)
    # Reorganize final output ------
    tem=1
    if wwl==1:
        pack_output_wwl(tem,rdir)
    else:
        pack_output_nl(tem,rdir)


def load_var(inv,infile):
    if os.path.exists(infile):
        inv=infile
        return 1,inv
    else:
        return 0,inv

def scan_input_train_mode(indir,disease):
    input_fs=''
    eg_fs=''
    eg_fs_norm=''
    meta=''
    insp=''
    check1,input_fs=load_var(input_fs,indir+'/'+disease+'_sp_train_norm_node.csv')
    check2,eg_fs=load_var(eg_fs,indir+'/'+disease+'_train_eggNOG_raw.csv')
    check3,eg_fs_norm=load_var(eg_fs_norm,indir+'/'+disease+'_train_eggNOG_norm.csv')
    check4,meta=load_var(meta,indir+'/'+disease+'_meta.tsv')
    check5,insp=load_var(insp,indir+'/'+disease+'_train_sp_norm.csv')
    check=check1+check2+check3+check4+check5
    if not check==5:
        print('Some input files are not provided, check please!')
        exit()
    pre_features={}
    if not os.path.exists(indir+'/pre_features'):
        print('Can not find the dir of pre-selected features, will re-select features!')
    for filename in os.listdir(indir+'/pre_features'):
        pre=re.split('_',filename)[0]
        pre=re.sub('Fold','',pre)
        pre=int(pre)
        fp=open(indir+'/pre_features/'+filename,'r')
        pre_features[pre]=indir+'/pre_features/'+filename
    return input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features


def scan_input(indir,disease):
    input_fs=''
    eg_fs=''
    eg_fs_norm=''
    meta=''
    insp=''
    check1,input_fs=load_var(input_fs,indir+'/'+disease+'_sp_merge_norm_node.csv')
    check2,eg_fs=load_var(eg_fs,indir+'/'+disease+'_eggNOG_merge_raw.csv')
    check3,eg_fs_norm=load_var(eg_fs_norm,indir+'/'+disease+'_eggNOG_merge_norm.csv')
    check4,meta=load_var(meta,indir+'/'+disease+'_meta.tsv')
    check5,insp=load_var(insp,indir+'/'+disease+'_sp_merge_norm.csv')
    check= check1+check2+check3+check4+check5
    if not check==5:
        print('Some input files are not provided, check please!')
        exit()
    # Check whether features are pre-selected.
    print('Scan whether the pre-selected features available...')
    pre_features={}
    if not os.path.exists(indir+'/kfold_features'):
        print('Can not find the dir of pre-selected features, will re-select features!')
    for filename in os.listdir(indir+'/pre_features'):
        pre=re.split('_',filename)[0]
        pre=re.sub('Fold','',pre)
        pre=int(pre)
        fp=open(indir+'/pre_features/'+filename,'r')
        pre_features[pre]=indir+'/pre_features/'+filename

    
    return input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features


    

def main():
    usage="GDmicro - Use GCN and domain adaptation to predict disease based on microbiome data."
    parser=argparse.ArgumentParser(prog="GDmicro.py",description=usage)
    parser.add_argument('-i','--input_dir',dest='input_dir',type=str,help="The directory of all input datasets. Should be the output of GDmicro_preprocess")
    parser.add_argument('-t','--train_mode',dest='train_mode',type=str,help="If set to 1, then will apply k-fold cross validation to all input datasets. This mode can only be used when input datasets are all training data. The input data should be the output of the train mode of GDmicro_preprocess. (default: 0)")
    #parser.add_argument('-v','--close_cv',dest='close_cv',type=str,help="If set to 1, will close the k-fold cross-validation and use all datasets for training. Only work when \"train mode\" is off (-t 0). (default: 0)")
    
    parser.add_argument('-d','--disease',dest='disease',type=str,help="The name of the disease.")
    parser.add_argument('-k','--kneighbor',dest='kneighbor',type=str,help="The number of neighborhoods in the knn graph. (default: 5)")
    parser.add_argument('-e','--apply_node',dest='anode',type=str,help="If set to 1, then will apply node importance calculation, which may take a long time. (default: not use).")
    parser.add_argument('-n','--node_num',dest='nnum',type=str,help="How many nodes will be output during the node importance calculation process. (default:20).")
    #parser.add_argument('-f','--feature_num',dest='fnum',type=str,help="How many features will be output during the feature importance calculation process. (default:20)")
    parser.add_argument('-c','--cvfold',dest='cvfold',type=str,help="The value of k in k-fold cross validation.  (default: 10)")
    parser.add_argument('-s','--randomseed',dest='rseed',type=str,help="The random seed used to reproduce the result.  (default: not use)")
    parser.add_argument('-a','--domain_adapt',dest='doadpt',type=str,help="Whether apply domain adaptation to the test dataset. If set to 0, then will use MLP rather than domain adaptation. (default: use)")

    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of test results. (Default: GDmicro_res)")

    args=parser.parse_args()
    indir=args.input_dir
    train_mode=args.train_mode
    #close_cv=args.close_cv
    #input_fs=args.input_fs
    #eg_fs=args.eg_fs
    #eg_fs_norm=args.eg_fs_norm
    #meta=args.meta
    anode=args.anode
    disease=args.disease
    nnum=args.nnum
    #fnum=args.fnum
    kneighbor=args.kneighbor
    #fuse=args.fuse
    cvfold=args.cvfold
    rseed=args.rseed
    doadpt=args.doadpt

    out=args.outdir
    close_cv=0
    fnum=100
    '''
    if not close_cv:
        close_cv=0
    else:
        close_cv=int(close_cv)
    '''
    if not anode:
        anode=0
    else:
        anode=int(anode)
    if not nnum:
        nnum=20
    else:
        nnum=int(nnum)
    '''
    if not fnum:
        fnum=20
    else:
        fnum=int(fnum)
    '''
    if not kneighbor:
        kneighbor=5
    else:
        kneighbor=int(kneighbor)
    if not train_mode:
        train_mode=0
    else:
        train_mode=int(train_mode)
    if not cvfold:
        cvfold=10
    else:
        cvfold=int(cvfold)
    if not rseed:
        rseed=0
    else:
        rseed=int(rseed)
    if not doadpt:
        doadpt=1
    else:
        doadpt=int(doadpt)
    if not out:
        out="GDmicro_res"
    

    
    if train_mode==0:
        input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features=scan_input(indir,disease)
        run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,pre_features,rseed,cvfold,doadpt,insp,fnum,nnum,close_cv,anode)
    else:
        input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features=scan_input_train_mode(indir,disease)
        run_GCN_train_mode.run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,rseed,cvfold,insp,fnum,nnum,pre_features,anode)




if __name__=="__main__":
    sys.exit(main())
