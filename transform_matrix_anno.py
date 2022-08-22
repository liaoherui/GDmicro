import re
import os

def trans(inmatrix,odir,pre,insample):
    f1=open(insample,'r')
    line=f1.readline()
    d={}
    while True:
        line=f1.readline().strip()
        if not line:break
        ele=line.split('\t')
        if ele[3]=='healthy':
            d['S'+ele[0]]='Health'
        else:
            d['S'+ele[0]]=ele[3]
    f2=open(inmatrix,'r')
    o=open(odir+'/'+pre+'_different_nf_value.txt','w+')
    while True:
        line=f2.readline().strip()
        if not line:break
        ele=re.split(',',line)
        name=ele[0]
        ele[0]=re.sub('S','',ele[0])
        o.write('\t'.join(ele)+'\t'+d[name]+'\n')

