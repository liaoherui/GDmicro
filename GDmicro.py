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
import run_MLP_embedding_da_for_node
import run_MLP_embedding
from GDmicro_preprocess import preprocess
from gcn_model import encode_onehot,GCN,train,train_fs,test,test_new_acc,test_unknown,normalize,sparse_mx_to_torch_sparse_tensor
from calculate_avg_acc_of_cross_validation_test import cal_acc_cv
import calculate_avg_acc_of_cross_validation_train_mode
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
    #print(train_idx)
    #exit()
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
def iter_run(features,train_id,test_id , adj, labels, ot2, rdir,classes_dict, tid2name, wwl,close_cv):
    model = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # val_idx=test_id
    max_train_auc = 0
    for epoch in range(150):
        train_auc, train_prob = train_fs(epoch, np.array(train_id), np.array(test_id), model, optimizer, features, adj, labels, ot2, max_train_auc, rdir, 0, classes_dict, tid2name, wwl, 0, close_cv)
        if wwl == 1:
            train_auc = float(train_auc)
            if train_auc > max_train_auc:
                max_train_auc = train_auc
                best_prob = train_prob
        else:
            train_auc = float(train_auc)
            if train_auc > max_train_auc:
                max_train_auc = train_auc
                best_prob = train_prob
    return best_prob

def detect_dsp(graph, eg_fs_norm,feature_id, labels_raw,labels,adj, train_id, test_id, rdir,ot2,classes_dict, tid2name, wwl,close_cv,sid,sname):
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
        train_auc, train_prob = train_fs(epoch, np.array(train_id), np.array(test_id), model, optimizer, features,adj, labels, ot2, max_train_auc, rdir, 0, classes_dict, tid2name, wwl, 0,close_cv)
        if wwl == 1:
            train_auc = float(train_auc)
            if train_auc > max_train_auc:
                max_train_auc = train_auc
                best_prob = train_prob
        else:
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


            bp1 = iter_run(features_one, train_id,test_id, adj, labels, ot2, rdir,classes_dict, tid2name, wwl,close_cv)
            bp2 = iter_run(features_two, train_id, test_id, adj, labels, ot2, rdir, classes_dict, tid2name, wwl,close_cv)
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
    o=open(rdir+'/driver_sp_stat.txt','w+')
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

    o2=open(rdir+'/driver_sp_change.txt','w+')
    o2.write('Species_ID\tSpecies_name\tIncrease2Disease\tIncrease2Health\tDecrease2Disease\tDecrease2Health\tValid_s\n')
    c=1
    for s in sname:
        o2.write(str(c)+'\t'+s+'\t'+str(avc[s]['Increase2Disease'])+'\t'+str(avc[s]['Increase2Health'])+'\t'+str(avc[s]['Decrease2Disease'])+'\t'+str(avc[s]['Decrease2Health'])+'\t'+str(vnsa[s])+'\n')
        c+=1










def feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,wwl,ot,dcs,fnum,close_cv,o4,eg_fs_norm):
    setup_seed(10)
    cround=1
    top100={}
    idx_features_labels = np.genfromtxt("{}".format(eg_fs_norm), dtype=np.dtype(str))
    features_raw = idx_features_labels[:, 1:-1]
    features_raw = features_raw.astype(float)
    features_raw = np.array(features_raw)
    #print(feature_id)
    #exit()
    while True:
        res={}
        prob_matrix=[]
        if cround==2:break
        for i in feature_id:
            max_test_auc=0
            max_train_auc=0
            best_prob=[]
            i=int(i)
            #if not i==597:continue
            if i in selected:continue
            features_tem=[[x[i]] for x in features]
            features_tem=torch.Tensor(features_tem)
            model=GCN(nfeat=features_tem.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                train_auc,train_prob=train_fs(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features_tem,adj,labels,ot,max_train_auc,rdir,fn+1,classes_dict,tid2name,wwl,0,close_cv)
                train_auc=float(train_auc)
                if wwl==1:
                    test_auc=test(model,test_idx,features_tem,adj,labels,ot,max_test_auc,rdir,fn+1,classes_dict,tid2name,0)
                    test_auc=float(test_auc)
                    if train_auc>max_train_auc:
                        max_train_auc=train_auc
                        best_prob=train_prob
                    if test_auc>float(max_test_auc):
                        max_test_auc=float(test_auc)
                else:
                    train_auc=float(train_auc)
                    if train_auc>max_train_auc:
                        max_train_auc=train_auc
                        best_prob=train_prob
            if wwl==1:
                res[i]=float(max_train_auc)
                prob_matrix.append(best_prob[:,1])
            else:
                res[i]=float(max_train_auc)
                prob_matrix.append(best_prob[:,1])
            train_v=features_raw[train_idx]
            train_v=np.sum(train_v[:,i])
            val_v=features_raw[val_idx]
            val_v= np.sum(val_v[:, i])
            test_v=features_raw[test_idx]
            test_sum=np.sum(test_v[:, i])
            train_sum=train_v+val_v
            '''
            a=features_raw[train_idx]
            a=a[:,i]
            b=features_raw[val_idx]
            b=b[:,i]
            c=features_raw[test_idx]
            c=c[:,i]
            print('Index',i)
            print(a,b,c)
            print(train_sum,test_sum)
            exit()
            '''
            if train_sum==0 or test_sum==0:
                res[i] = float(0)

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

        prob_matrix=np.array(prob_matrix).T
        savetxt(o4,prob_matrix,delimiter=',')

    sid=sorted(list(top100.keys()))
    sname=[]
    for s in sid:
        sname.append(top100[s])
    return sid,sname

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
        #o.write('\n')
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

def run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,pre_features,rseed,cvfold,doadpt,insp,fnum,nnum,close_cv,anode,reverse,vnode,uf,bsize,rfi):
    if not rseed==0:
        setup_seed(rseed)
    # Load species name -> for feature importance
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
    
    if vnode==0:
        idx_features_labels = np.genfromtxt("{}".format(input_fs),dtype=np.dtype(str))
    else:
        new_fs=run_MLP_embedding_da_for_node.
    '''
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

    btgraph=''
    btauc=0
    for train_idx,val_idx in datasets:
        #print(train_idx)
        #exit()
        #o3=open(rdir+'/sample_prob_fold'+str(fn+1)+'.txt','w+')
        # Select features using lasso
        o1.write('Fold {}'.format(fn + 1) + '\n')
        
        # Select features using lasso
        if uf==0:
            if len(pre_features)==0:
                eg_fs_sf=select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn+1)
            else:
                if len(pre_features)==cvfold:
                    eg_fs_sf=pre_features[fn+1]
                else:
                    eg_fs_sf=select_features(eg_fs,eg_fs_norm,train_idx,fdir,meta,disease,fn+1)
             
        # If reverse, then transfer eggNOG features to node features
        if reverse==1:
            otem=uuid.uuid1().hex+'.csv'
            eg_node=trans_node(eg_fs_sf,meta,otem)
            idx_features_labels = np.genfromtxt("{}".format(otem),dtype=np.dtype(str))
            features=idx_features_labels[:, 1:-1]
            features=features.astype(float)
            features=np.array(features)
            if vnode==0:
                os.system('rm '+otem)
            dcs={}
            load_dcs(eg_fs_sf,dcs)




        #eg_fs_sf='CRC_41_GCN/FRA_k10/Feature_File/eggNOG_features_Fold1.tsv'
        # Usa all features
        '''
        eg_fs_sf=eg_fs_norm
        '''
        # Train MLP on selected features 10 times and selecte the best model to build the graph
        #exit()
        #graph=run_MLP_embedding.build_graph_mlp('../New_datasets/T2D_data_2012_Trans/T2D_eggNOG_norm.txt',train_idx,val_idx,meta,disease,fn+1,gdir)
        if doadpt==1:
            if reverse==0 and uf==0:
                graph=run_MLP_embedding_da.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,bsize)
            else:
                graph=run_MLP_embedding_da.build_graph_mlp(insp,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,bsize)
        else:
            if reverse==0 and uf==0:
                graph=run_MLP_embedding.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,bsize)
            else:
                graph=run_MLP_embedding.build_graph_mlp(insp,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,bsize)

        #exit()
        

        # Train and testing 
        labels,classes_dict=encode_onehot(labels_raw)
        #print(classes_dict)
        #exit()
        
        if vnode==0:
            features = sp.csr_matrix(features, dtype=np.float32)
        else:
            if reverse==0:
                embd_vector=run_MLP_embedding_da_for_node.build_graph_mlp(insp,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,input_fs)

            else:
                embd_vector=run_MLP_embedding_da_for_node.build_graph_mlp(eg_fs_sf,train_idx,val_idx,meta,disease,fn+1,gdir,test_idx,kneighbor,rseed,wwl,rdir,close_cv,otem)
                os.system('rm '+otem)
            idx_features_labels = np.genfromtxt("{}".format(embd_vector),dtype=np.dtype(str))
            features=idx_features_labels[:, 1:-1]
            features=features.astype(float)
            features=np.array(features)
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
        #print(features)
        labels_copy=labels.clone().detach()
        #exit()
        model=GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max().item() + 1, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
        max_val_auc=0
        max_test_auc=0
        #max_test_acc=0
        for epoch in range(150):
            val_auc=train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,o1,max_val_auc,rdir,fn+1,classes_dict,tid2name,wwl,1,close_cv)
            raw_mval_auc=max_val_auc
            if val_auc>max_val_auc:
                max_val_auc=val_auc
            ### New part for testing datasets
            if wwl==1 and close_cv==0:
                if len(test_idx)<13:
                    test_auc=test_new_acc(model,test_idx,features,adj,labels,o1,max_test_auc,rdir,fn+1,classes_dict,tid2name,1)
                else:
                    test_auc=test(model,test_idx,features,adj,labels,o1,max_test_auc,rdir,fn+1,classes_dict,tid2name,1)
                if test_auc>max_test_auc:
                    max_test_auc=test_auc
                if test_auc>btauc:
                    btauc=test_auc
                    btgraph=graph
                    bset=[]
                    bset.append(train_idx)
                    bset.append(val_idx)
                    bset.append(labels_copy)
                    bset.append(adj)

                    bset.append(fn)
            else:
                if wwl==1:
                    if len(test_idx)<13:
                        test_auc=test_new_acc(model,test_idx,features,adj,labels,o1,max_test_auc,rdir,fn+1,classes_dict,tid2name,1)
                    else:
                        test_auc=test(model,test_idx,features,adj,labels,o1,max_test_auc,rdir,fn+1,classes_dict,tid2name,1)
                    if test_auc>max_test_auc:
                        max_test_auc=test_auc
                    if test_auc>btauc:
                        btauc = test_auc
                        btgraph = graph
                        bset = []
                        bset.append(train_idx)
                        bset.append(val_idx)
                        bset.append(labels_copy)
                        bset.append(adj)

                        bset.append(fn)
                else:
                    if val_auc>raw_mval_auc:
                        test_unknown(model,test_idx,features,adj,rdir,fn+1,classes_dict,tid2name,1)
                    if val_auc>btauc:
                        btauc= val_auc
                        btgraph = graph
                        bset = []
                        bset.append(train_idx)
                        bset.append(val_idx)
                        bset.append(labels_copy)
                        bset.append(adj)

                        bset.append(fn)
        
        ##### Feature importance
        # if vnode==100:
        #     selected={}
        #     selected_arr=[]
        #     o3=open(rdir+'/feature_importance_fold'+str(fn+1)+'.txt','w+')
        #     #o4=open(rdir+'/feature_importance_iterative_fold'+str(fn+1)+'.txt','w+')
        #     o4=open(rdir+'/feature_local_importance_fold'+str(fn+1)+'.txt','w+')
        #     uid=uuid.uuid1().hex
        #     ot=open(uid+'.log','w+')
        #     feature_importance_check(selected,selected_arr,feature_id,train_idx,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o3,wwl,ot,dcs,fnum,close_cv,o4)
        #     ot.close()
        #     os.system('rm '+uid+'.log')
        

        ##### Node importance
        # if anode==1:
        #     selected={}
        #     selected_arr=[]
        #     o5=open(rdir+'/node_importance_single_fold'+str(fn+1)+'.txt','w+')
        #     o6=open(rdir+'/node_importance_combination_fold'+str(fn+1)+'.txt','w+')
        #     uid=uuid.uuid1().hex
        #     ot2=open(uid+'.log','w+')
        #     node_importance_check(selected,selected_arr,tem_train_id,val_idx,test_idx,features,adj,labels,rdir,fn,classes_dict,tid2name,o5,o6,wwl,ot2,nnum,close_cv)
        #     ot2.close()
        #     os.system('rm '+uid+'.log')
        

        fn+=1
        #break

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
    feature_id=list(range(int(features.shape[1])))
    # Feature importance
    
    if rfi==1:
        selected = {}
        selected_arr = []
        o3 = open(rdir + '/feature_importance.txt', 'w+')
        # o4=open(rdir+'/feature_importance_iterative_fold'+str(fn+1)+'.txt','w+')
        o4 = open(rdir + '/feature_local_importance.txt', 'w+')
        uid = uuid.uuid1().hex
        ot = open(uid + '.log', 'w+')
        sid,sname=feature_importance_check(selected, selected_arr, feature_id, bset[0], bset[1], test_idx, features, bset[3],bset[2], rdir, bset[4], classes_dict, tid2name, o3, wwl, ot, dcs, fnum, close_cv,o4,eg_fs_norm)
        ot.close()
        os.system('rm ' + uid + '.log')
        #### Biomarker influence score
        uid = uuid.uuid1().hex
        ot2 = open(uid + '.log', 'w+')
        detect_dsp(btgraph, eg_fs_norm,feature_id, labels_raw,bset[2],bset[3], tem_train_id, test_idx, rdir,ot2,classes_dict, tid2name, wwl,close_cv,sid,sname)
        ot2.close()
        os.system('rm ' + uid + '.log')

    os.system('rm -rf '+rdir+'/tem_files')
    os.system('rm -rf '+fdir)
    os.system('rm -rf '+gdir)
    #exit()



def load_var(inv,infile):
    if os.path.exists(infile):
        inv=infile
        return 1,inv
    else:
        return 0,inv

def scan_input_train_mode(indir,disease,uf):
    input_fs=''
    eg_fs=''
    eg_fs_norm=''
    meta=''
    insp=''
    check1,input_fs=load_var(input_fs,indir+'/'+disease+'_sp_train_norm_node.csv')
    check2,eg_fs=load_var(eg_fs,indir+'/'+disease+'_train_sp_raw.csv')
    check3,eg_fs_norm=load_var(eg_fs_norm,indir+'/'+disease+'_sp_train_raw_node.csv')
    check4,meta=load_var(meta,indir+'/'+disease+'_meta.tsv')
    check5,insp=load_var(insp,indir+'/'+disease+'_train_sp_norm.csv')
    check=check1+check2+check3+check4+check5
    if not check==5 and uf==0:
        print('Some input files are not provided, check please!')
        exit()
    pre_features={}
    if not os.path.exists(indir+'/pre_features'):
        #print('Can not find the dir of pre-selected features, will re-select features!')
        x=1
    else:
        for filename in os.listdir(indir+'/pre_features'):
            pre=re.split('_',filename)[0]
            pre=re.sub('Fold','',pre)
            pre=int(pre)
            fp=open(indir+'/pre_features/'+filename,'r')
            pre_features[pre]=indir+'/pre_features/'+filename
    return input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features


def scan_input(indir,disease,uf):
    input_fs=''
    eg_fs=''
    eg_fs_norm=''
    meta=''
    insp=''
    check1,input_fs=load_var(input_fs,indir+'/'+disease+'_sp_merge_norm_node.csv')
    check2,eg_fs=load_var(eg_fs,indir+'/'+disease+'_sp_merge_raw.csv')
    check3,eg_fs_norm=load_var(eg_fs_norm,indir+'/'+disease+'_sp_merge_raw_node.csv')
    check4,meta=load_var(meta,indir+'/'+disease+'_meta.tsv')
    check5,insp=load_var(insp,indir+'/'+disease+'_sp_merge_norm.csv')
    check= check1+check2+check3+check4+check5
    if not check==5 and uf==0:
        print('Some input files are not provided, check please!')
        exit()
    # Check whether features are pre-selected.
    #print('Scan whether the pre-selected features available...')
    pre_features={}
    if not os.path.exists(indir+'/pre_features'):
        #print('Can not find the dir of pre-selected features, will re-select features!')
        x=1
    else:
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
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input csv file.")
    parser.add_argument('-t','--train_mode',dest='train_mode',type=str,help="If set to 1, then will apply k-fold cross validation to all input datasets. This mode can only be used when input datasets all have labels and set as \"train\" in input file.")
    #parser.add_argument('-v','--close_cv',dest='close_cv',type=str,help="If set to 1, will close the k-fold cross-validation and use all datasets for training. Only work when \"train mode\" is off (-t 0). (default: 0)")
    
    parser.add_argument('-d','--disease',dest='disease',type=str,help="The name of the disease.")
    parser.add_argument('-k','--kneighbor',dest='kneighbor',type=str,help="The number of neighborhoods in the knn graph. (default: 5)")
    parser.add_argument('-b','--batchsize',dest='bsize',type=str,help="The batch size during the training process. Should be set to 1 if there is only one test sample. (default: 64)")
    parser.add_argument('-e','--apply_node',dest='anode',type=str,help="If set to 1, then will apply node importance calculation, which may take a long time. (default: not use).")
    parser.add_argument('-n','--node_num',dest='nnum',type=str,help="How many nodes will be output during the node importance calculation process. (default:20).")
    parser.add_argument('-f','--feature_num',dest='fnum',type=str,help="How many features (top x features) will be analyzed during the feature influence score calculation process. (default: x=10)")
    parser.add_argument('-c','--cvfold',dest='cvfold',type=str,help="The value of k in k-fold cross validation.  (default: 10)")
    parser.add_argument('-s','--randomseed',dest='rseed',type=str,help="The random seed used to reproduce the result.  (default: not use)")
    parser.add_argument('-a','--domain_adapt',dest='doadpt',type=str,help="Whether apply domain adaptation to the test dataset. If set to 0, then will use MLP rather than domain adaptation. (default: use)")
    parser.add_argument('-r','--run_fi',dest='rfi',type=str,help="Whether run feature importance calculation process. If set to 0, then will not calculate the feature importance and contribution score. (default: 1)")
    #parser.add_argument('-r','--reverse',dest='reverse',type=str,help="If set to 1, then will use functional data as node features, and compostitional data to build edges. (default: 0)")
    #parser.add_argument('-v','--embed_vector_node',dest='vnode',type=str,help="If set to 1, then will apply domain adaptation network to node features, and use embedding vectors as nodes.. (default: 0)")
    #parser.add_argument('-u','--unique_feature',dest='uf',type=str,help="If set to 1, then will only use compostitional data to build edges and as node features.")

    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of test results. (Default: GDmicro_res)")

    args=parser.parse_args()
    infile=args.input_file
    train_mode=args.train_mode
    bsize=args.bsize
    rfi=args.rfi

    #close_cv=args.close_cv
    #input_fs=args.input_fs
    #eg_fs=args.eg_fs
    #eg_fs_norm=args.eg_fs_norm
    #meta=args.meta
    anode=args.anode
    disease=args.disease
    nnum=args.nnum
    fnum=args.fnum
    kneighbor=args.kneighbor
    #fuse=args.fuse
    cvfold=args.cvfold
    reverse=0
    vnode=0
    uf=1
    rseed=args.rseed
    doadpt=args.doadpt

    out=args.outdir
    close_cv=0
    #fnum=100
    '''
    if not close_cv:
        close_cv=0
    else:
        close_cv=int(close_cv)
    
    if not reverse:
        reverse=0
    else:
        reverse=int(reverse)
    if not uf:
        uf=1
    else:
        uf=int(uf)
    if not vnode:
        vnode=0
    else:
        vnode=int(vnode)
    '''
    if not bsize:
        bsize=64
    else:
        bsize=int(bsize)
    if not rfi:
        rfi=1
    else:
        rfi=int(rfi)
    if not anode:
        anode=0
    else:
        anode=int(anode)
    if not nnum:
        nnum=20
    else:
        nnum=int(nnum)
    
    if not fnum:
        fnum=10
    else:
        fnum=int(fnum)
    
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
    
    indir=preprocess(infile,train_mode,disease,out)
    #print(indir)
    #exit()
    
    if train_mode==0:
        input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features=scan_input(indir,disease,uf)
        run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,pre_features,rseed,cvfold,doadpt,insp,fnum,nnum,close_cv,anode,reverse,vnode,uf,bsize,rfi)
    else:
        input_fs,eg_fs,eg_fs_norm,meta,insp,pre_features=scan_input_train_mode(indir,disease,uf)
        run_GCN_train_mode.run(input_fs,eg_fs,eg_fs_norm,meta,disease,out,kneighbor,rseed,cvfold,insp,fnum,nnum,pre_features,anode,reverse,uf)




if __name__=="__main__":
    sys.exit(main())
