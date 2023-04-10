import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import run_MLP_embedding_train_mode
from gcn_model_train_mode import encode_onehot,GCN,train,train_fs,test,normalize,sparse_mx_to_torch_sparse_tensor
from calculate_avg_acc_of_cross_validation_train_mode import cal_acc_cv
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import random
import uuid
from numpy import savetxt

#exit()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

def avg_score(avc,vnsa):
    for s in avc:
        if vnsa[s]==0:
            avc[s]['Increase2Disease'] =0
            avc[s]['Increase2Health'] = 0
            avc[s]['Decrease2Disease'] = 0
            avc[s]['Decrease2Health'] = 0
        else:
            avc[s]['Increase2Disease']=sum(avc[s]['Increase2Disease'])/vnsa[s]
            avc[s]['Increase2Health']=sum(avc[s]['Increase2Health'])/vnsa[s]
            avc[s]['Decrease2Disease'] = sum(avc[s]['Decrease2Disease']) / vnsa[s]
            avc[s]['Decrease2Health'] = sum(avc[s]['Decrease2Health']) / vnsa[s]
    return avc

def iter_run(features,train_id,test_id , adj, labels, ot2, rdir,classes_dict, tid2name):

    model = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # val_idx=test_id
    max_train_auc = 0
    for epoch in range(150):
        train_auc, train_prob = train_fs(epoch, np.array(train_id), np.array(test_id), model, optimizer, features, adj, labels, ot2, max_train_auc, rdir, 0, classes_dict, tid2name,  0)
        train_auc = float(train_auc)
        if train_auc > max_train_auc:
            max_train_auc = train_auc
            best_prob = train_prob

    return best_prob


def detect_dsp(graph, eg_fs_norm,feature_id, labels_raw,labels,adj, train_id, test_id, rdir,ot2,classes_dict, tid2name,sid,sname,fn):
    wwl=1
    close_cv=0
    setup_seed(10)
    dn={}
    idx_features_labels = np.genfromtxt("{}".format(eg_fs_norm), dtype=np.dtype(str))
    features = idx_features_labels[:, 1:-1]
    features = features.astype(float)
    features = np.array(features)

    features_raw=features.copy()
    features = sp.csr_matrix(features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    #print(feature_id,graph)
    feature_id = list(range(int(features.shape[1])))
    #print(feature_id)
    #exit()


    f=open(graph,'r')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        ele[0]=int(ele[0])
        ele[1]=int(ele[1])
        if ele[0] not in dn:
            dn[ele[0]]={ele[1]:''}
        else:
            dn[ele[0]][ele[1]]=''
        if ele[1] not in dn:
            dn[ele[1]]={ele[0]:''}
        else:
            dn[ele[1]][ele[0]]=''
    tg=[] # only consider training data for now
    for s in dn:
        p=0
        n=0
        if s not in train_id:continue
        for s2 in dn[s]:
            if s2 not in train_id:continue
            if labels_raw[s2]=='Health':
                n+=1
            else:
                p+=1
        if p>=0 and n>=0:
            tg.append(s)
    print('There are '+str(len(tg))+' samples have both >=0 healthy and disease neighbors.')
    #print(features)
    #print(features.shape[1])
    #print(labels)
    #exit()
    model = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # val_idx=test_id
    # Only consider correctly identified samples
    max_train_auc=0
    for epoch in range(150):
        print('DSD_Raw')
        train_auc, train_prob = train_fs(epoch, np.array(train_id), np.array(test_id), model, optimizer, features,adj, labels, ot2, max_train_auc, rdir, 0, classes_dict, tid2name, 0)
        train_auc = float(train_auc)
        if train_auc > max_train_auc:
            max_train_auc = train_auc
            best_prob = train_prob

    tgc=[]
    for t in tg:
        if best_prob[t][0]>best_prob[t][1]:
            prl=0
        else:
            prl=1
        if prl==labels[t]:
            tgc.append(t)

    print(len(tgc),' samples will be used to detect driver species...')
    #exit()
    # there are several types:
    # Old rule:Raw: 0: raw_prob, Max: 1: Max2Median, 2: Max2Zero, Middle: 3: Middle2Max, 4: Middle2Zero, Zero: 5: Zero2Median, 6: Zero2Max
    # New rule: Raw: 0: raw_prob, Max: 1: Max2Median, 2: Max2Min, Middle: 3: Middle2Max, 4: Middle2Min, Min: 5: Min2Median, 6: Min2Max

    res={} # sample_id -> feature_id -> [0.55,-1,-1,0.52,0.33,-1,-1]
    arr=[] # sample_id list
    for t  in tgc:
        arr.append(t)
        tem_train=train_id.copy()
        # tem_train.remove(t)
        raw_prob=best_prob[t][1]
        c_feature=features_raw.copy()
        t_feature=c_feature[t]
        ab_max=np.max(t_feature)
        ab_median=np.median(t_feature)
        ab_min=np.min(t_feature)
        res[t]={}
        for s in feature_id:
            if s not in sid:continue
            res[t][s]=['-1','-1','-1','-1','-1','-1','-1']
            res[t][s][0]=str(raw_prob)
            raw_feature_value=t_feature[s]
            if float(raw_feature_value)==0:continue
            set_index=[]
            if float(raw_feature_value)==ab_min:
                features_one=features.clone().detach()
                features_one[t][s]=ab_median
                features_two=features.clone().detach()
                features_two[t][s]=ab_max
                set_index.append(5)
                set_index.append(6)
            elif float(raw_feature_value)==ab_max:
                features_one =features.clone().detach()
                features_one[t][s]=ab_median
                features_two =features.clone().detach()
                features_two[t][s]=ab_min
                set_index.append(1)
                set_index.append(2)
            else:
                features_one =features.clone().detach()
                features_one[t][s] = ab_max
                features_two =features.clone().detach()
                features_two[t][s] = ab_min
                set_index.append(3)
                set_index.append(4)


            bp1 = iter_run(features_one, train_id,test_id, adj, labels, ot2, rdir,classes_dict, tid2name)
            bp2 = iter_run(features_two, train_id, test_id, adj, labels, ot2, rdir, classes_dict, tid2name)
            res[t][s][set_index[0]]= str(bp1[t][1])
            res[t][s][set_index[1]] = str(bp2[t][1])
    disease_lab=0
    health_lab=0
    for c in classes_dict:
        if c=='Health':
            if classes_dict['Health'][0]==1:
                health_lab=0
                disease_lab=1
            else:
                health_lab = 1
                disease_lab = 0



    # Increase abundance (3, 5, 6) -> close to CRC or close to Health | Decrease abundance (1, 2, 4) -> close to CRC or close to Health
    # Raw: 0: raw_prob, Max: 1: Max2Median, 2: Max2Zero, Middle: 3: Middle2Max, 4: Middle2Zero, Zero: 5: Zero2Median, 6: Zero2Max
    # New rule: Raw: 0: raw_prob, Max: 1: Max2Median, 2: Max2Min, Middle: 3: Middle2Max, 4: Middle2Min, Min: 5: Min2Median, 6: Min2Max
    o=open(rdir+'/driver_sp_stat_fold'+str(fn+1)+'.txt','w+')
    iab=[3,5,6]
    dab=[1,2,4]
    avc={} # Calculate average change of each feature across disease and healthy samples
            # feature_name-> Disease: change_value | Health: change_value
    sp_name = dict(zip(sid, sname))

    o.write('Sample_ID\tLabel\t'+'\t'.join(sname)+'\n')

    # Calculate valid samples
    vsa={} # sid -> valid sample
    vnsa={} # sname -> valid sample
    for t in res:
        #valid=0
        for s in feature_id:
            valid=0
            if s not in sid: continue
            if s not in vsa:
                vsa[s]=0
                vnsa[sp_name[s]]=0
            if not res[t][s][1] == '-1':
                c1 = float(res[t][s][0]) - float(res[t][s][1])
                c2 = float(res[t][s][0]) - float(res[t][s][2])
                if c1 < 0 and c2 < 0 and abs(c1) < abs(c2):valid=1
                if c1 > 0 and c2 > 0 and c1 < c2:valid=1
            elif not res[t][s][3] == '-1':
                c3 = float(res[t][s][0]) - float(res[t][s][3])
                c4 = float(res[t][s][0]) - float(res[t][s][4])
                if c3 > 0 and c4 < 0:valid=1
                if c3 < 0 and c4 > 0:valid=1
            elif not res[t][s][5] == '-1':
                c5 = float(res[t][s][0]) - float(res[t][s][5])
                c6 = float(res[t][s][0]) - float(res[t][s][6])
                if c5 > 0 and c6 < 0 :valid=1
                if c5 < 0 and c6 > 0 :valid=1
            if valid==1:
                vsa[s]+=1
                vnsa[sp_name[s]]+=1


    for t in res:
        o.write(str(t)+'\t'+labels_raw[t]+'\t')
        tem=[]
        # tid=0
        for s in feature_id:
            if s not in sid: continue
            tem.append(','.join(res[t][s]))
            if sp_name[s] not in avc:
                avc[sp_name[s]] = {'Increase2Disease': [], 'Increase2Health':[], 'Decrease2Disease': [],
                                   'Decrease2Health': []}

            if not res[t][s][1] == '-1':
                c1 = float(res[t][s][0]) - float(res[t][s][1])
                c2 = float(res[t][s][0]) - float(res[t][s][2])

                if health_lab == 1:
                    if c1 < 0 and c2<0 and abs(c1)<abs(c2):
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c1)+abs(c2))
                    if c1>0 and c2>0 and c1<c2:
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c1)+abs(c2))

                else:
                    if c1 < 0 and c2<0 and abs(c1)<abs(c2):
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c1)+abs(c2))
                    if c1 > 0 and c2 > 0 and c1 < c2:
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c1)+abs(c2))

            elif not res[t][s][3] == '-1':
                c3 = float(res[t][s][0]) - float(res[t][s][3])
                c4 = float(res[t][s][0]) - float(res[t][s][4])
                if vsa[s]<15:
                    if abs(c3) > 0.3 or abs(c4) > 0.3: continue
                if health_lab == 1:
                    if c3 >0 and c4<0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c4))
                    if c3<0 and c4>0:
                        avc[sp_name[s]]['Increase2Health'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c4))
                else:
                    if c3 >0 and c4<0:
                        avc[sp_name[s]]['Increase2Health'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c4))
                    if c3 < 0 and c4 > 0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c4))


            elif not res[t][s][5] == '-1':
                c5 = float(res[t][s][0]) - float(res[t][s][5])
                c6 = float(res[t][s][0]) - float(res[t][s][6])
                if vsa[s]<15:
                    if abs(c3) > 0.3 or abs(c4) > 0.3: continue
                if health_lab == 1:
                    if c5 > 0 and c6<0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c5))
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c6))
                    if c5 <0 and c6 > 0:
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c5))
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c6))
                else:
                    if c5 > 0 and c6<0 and abs(c5)< abs(c6):
                        avc[sp_name[s]]['Increase2Health'].append(abs(c5))
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c6))

                    if c5 < 0 and c6 > 0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c5))
                        avc[sp_name[s]]['Increase2Health'].append(abs(c6))


        o.write('\t'.join(tem)+'\n')
    o.close()
    #raw_avc=avc.copy()
    avc=avg_score(avc,vnsa)

    o2=open(rdir+'/driver_sp_change_fold'+str(fn+1)+'.txt','w+')
    o2.write('Species_ID\tSpecies_name\tIncrease2Disease\tIncrease2Health\tDecrease2Disease\tDecrease2Health\tValid_s\n')
    c=1
    for s in sname:
        o2.write(str(c)+'\t'+s+'\t'+str(avc[s]['Increase2Disease'])+'\t'+str(avc[s]['Increase2Health'])+'\t'+str(avc[s]['Decrease2Disease'])+'\t'+str(avc[s]['Decrease2Health'])+'\t'+str(vnsa[s])+'\n')
        c+=1

def feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,ot,dcs,fnum,o4):
    setup_seed(10)
    cround=1
    top100={}
    while True:
        res={}
        prob_matrix = []
        if cround==2:break
        for i in feature_id:
            max_train_auc=0
            best_prob = []
            i=int(i)
            if i in selected:continue
            features_tem=[[x[i]] for x in features]
            features_tem=torch.Tensor(features_tem)
            model=GCN(nfeat=features_tem.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                train_auc,sample_prob=train_fs(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features_tem,adj,labels,ot,max_train_auc,rdir,fn+1,classes_dict,tid2name,0)
                train_auc=float(train_auc)
                if train_auc>max_train_auc:
                    max_train_auc=train_auc
                    best_prob = sample_prob
            res[i]=float(max_train_auc)
            prob_matrix.append(best_prob[:, 1])
        res2=sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        sid=1
        if cround==1:
            for r in res2:
                #if sid==fnum+1:break
                o3.write(str(sid)+'\t'+str(dcs[r[0]])+'\t'+str(r[1])+'\n')
                if sid<fnum+1:
                    top100[int(r[0])]=str(dcs[r[0]])
                sid+=1
            o3.close()
        selected[res2[0][0]]=res2[0][1]
        selected_arr.append(res2[0][0])
        cround+=1
        prob_matrix = np.array(prob_matrix).T
        savetxt(o4, prob_matrix, delimiter=',')
    sid=sorted(list(top100.keys()))
    sname=[]
    for s in sid:
        sname.append(top100[s])
    return sid,sname

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

def trans_node(infile,meta,ofile):
    f=open(meta,'r')
    line=f.readline()
    arr=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        arr.append(ele[3])
    a=pd.read_table(infile)
    a=np.array(a).T
    c=0
    o=open(ofile,'w+')
    for s in a:
        o.write(str(c))
        for e in s:
            o.write('\t'+str(e))
        o.write('\t'+arr[c]+'\n')
        c+=1
    o.close()

def load_dcs(infile,dcs):
    f=open(infile,'r')
    line=f.readline()
    cs=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        dcs[cs]=ele[0]
        cs+=1
 


def run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,rseed,cvfold,insp,fnum,nnum,pre_features,anode,reverse,uf):
    if not rseed==0:
        setup_seed(rseed)
    dcs={}
    load_dcs(insp,dcs)
    '''
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
    '''
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
        if uf==0:
            if len(pre_features)==0:
                eg_fs_sf=select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn+1)
            else:
                eg_fs_sf=pre_features[fn+1]

        if reverse==1:
            otem=uuid.uuid1().hex+'.csv'
            eg_node=trans_node(eg_fs_sf,meta,otem)
            idx_features_labels = np.genfromtxt("{}".format(otem),dtype=np.dtype(str))
            features=idx_features_labels[:, 1:-1]
            features=features.astype(float)
            features=np.array(features)
            os.system('rm '+otem)
            dcs={}
            load_dcs(eg_fs_sf,dcs)
        # Usa all features
        '''
        eg_fs_sf=eg_fs_norm
        '''
        # Train MLP on selected features 10 times and selecte the best model to build the graph
        #exit()
        #graph=run_MLP_embedding.build_graph_mlp('../New_datasets/T2D_data_2012_Trans/T2D_eggNOG_norm.txt',train_idx,val_idx,meta,disease,fn+1,gdir)
        if reverse==0 and uf==0:
            graph=run_MLP_embedding_train_mode.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,kneighbor,rseed,rdir)
        else:
            graph=run_MLP_embedding_train_mode.build_graph_mlp(insp,train_idx,val_idx,meta,disease,fn+1,gdir,kneighbor,rseed,rdir)
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
        o4 = open(rdir + '/feature_local_importance_fold' + str(fn + 1) + '.txt', 'w+')
        #o4=open(rdir+'/feature_importance_iterative_fold'+str(fn+1)+'.txt','w+')
        uid=uuid.uuid1().hex
        ot=open(uid+'.log','w+')
        sid,sname=feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,ot,dcs,fnum,o4)
        ot.close()
        os.system('rm '+uid+'.log')

        uid=uuid.uuid1().hex
        ot2=open(uid+'.log','w+')
        detect_dsp(graph,eg_fs_norm,feature_id,labels_raw,labels,adj,train_idx,val_idx,rdir,ot2,classes_dict,tid2name,sid,sname,fn)
        ot2.close()
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


        
    

# def main():
#     usage="Herui's test scripts."
#     parser=argparse.ArgumentParser(prog="run_GCN.py",description=usage)
#     parser.add_argument('-i','--input_feature',dest='input_fs',type=str,help="The input species feature file. (Node format)")
#     parser.add_argument('-e','--eggNOG_feature',dest='eg_fs',type=str,help="The input eggNOG feature file.")
#     parser.add_argument('-n','--eggNOG_feature_norm',dest='eg_fs_norm',type=str,help="The input normalized eggNOG feature file.")
#     parser.add_argument('-m','--metadata',dest='meta',type=str,help="The input metadata file.")
#     parser.add_argument('-d','--disease',dest='disease',type=str,help="The name of the disease.")
#     parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of test results. (Default: GCN_res")
#
#     args=parser.parse_args()
#
#     input_fs=args.input_fs
#     eg_fs=args.eg_fs
#     eg_fs_norm=args.eg_fs_norm
#     meta=args.meta
#     disease=args.disease
#     out=args.outdir
#
#     run(input_fs,eg_fs,eg_fs_norm,meta,disease,out)


'''
if __name__=="__main__":
    sys.exit(main())
'''
