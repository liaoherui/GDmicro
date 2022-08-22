import re
import os

def merge2(infile1,infile3,train_idx,test_idx,ofile):
    f1=open(infile1,'r')
    d={}
    c=0
    while True:
        line=f1.readline().strip()
        if not line:break
        d[train_idx[c]]=line
        c+=1
    f3=open(infile3,'r')
    c=0
    while True:
        line=f3.readline().strip()
        if not line:break
        d[test_idx[c]]=line
        c+=1
    o=open(ofile,'w+')
    total=len(train_idx)+len(test_idx)
    for c in range(total):
        o.write(d[c]+'\n')


def merge(infile1,infile2,infile3,train_idx,val_idx,test_idx,ofile):

    f1=open(infile1,'r')
    d={}
    c=0
    while True:
        line=f1.readline().strip()
        if not line:break
        d[train_idx[c]]=line
        c+=1
    f2=open(infile2,'r')
    #d2={}
    c=0
    while True:
        line=f2.readline().strip()
        if not line:break
        d[val_idx[c]]=line
        c+=1
    f3=open(infile3,'r')
    c=0
    while True:
        line=f3.readline().strip()
        if not line:break
        d[test_idx[c]]=line
        c+=1
    o=open(ofile,'w+')
    total=len(train_idx)+len(val_idx)+len(test_idx)
    for c in range(total):
        o.write(d[c]+'\n')
            

#merge('T2D_result/Graph_File/feature_out_train_Fold1_eggNOG.txt','T2D_result/Graph_File/feature_out_test_Fold1_eggNOG.txt',list(range(340)),list(range(340,363)),'T2D_result/Graph_File/merge_embedding_Fold1.txt')



        
