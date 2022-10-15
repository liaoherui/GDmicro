import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import uuid

def trans_meta(in1,in2,out):
    f1=open(in1,'r')
    f2=open(in2,'r')
    o=open(out,'w+')
    line=f1.readline().strip()
    o.write(line+'\tclass\n')
    c=0
    while True:
        line=f1.readline().strip()
        if not line:break
        o.write(line+'\ttrain\n')
        c+=1 
    line=f2.readline()
    while True:
        line=f2.readline().strip()
        if not line:break
        ele=line.split('\t')
        ele[0]=str(c)
        o.write('\t'.join(ele)+'\ttest\n')
        c+=1
    o.close()
def trans_meta_train(infile,out):
    f=open(infile,'r')
    o=open(out,'w+')
    line=f.readline().strip()
    o.write(line+'\tclass\n')
    while True:
        line=f.readline().strip()
        if not line:break
        o.write(line+'\ttrain\n')
    o.close()



def normalize_data(infile,mtype,meta,dtype,ofile):
    f=open(meta,'r')
    meta_content=[]
    line=f.readline().strip()
    meta_content.append(line)
    c=0
    ag=0
    while True:
        line=f.readline().strip()
        if not line:break
        meta_content.append(line)
        label=line.split('\t')[3]
        if label=='Unknown':
            ag=1
        c+=1
    if ag==1:
        n_split_d=int(c/2)
        n_split_h=c-n_split_d
        ml_d=[dtype for i in range(n_split_d)]
        ml_h=['healthy' for i in range(n_split_h)]
        ml=ml_d+ml_h
        #print(ml)
        #exit()
        uid=uuid.uuid1().hex
        tmeta='tem_meta_'+uid+'.tsv'
        ot=open(tmeta,'w+')
        ot.write(meta_content[0]+'\n')
        i=0
        for c in meta_content[1:]:
            ele=c.split('\t')
            ele[3]=ml[i]
            i+=1
            ot.write('\t'.join(ele)+'\n')
        ot.close()
        #exit()
        os.system('Rscript norm_features.R '+mtype+' '+infile+' '+tmeta+' '+dtype+' '+ofile)
        os.system('rm '+tmeta)
    else:
        os.system('Rscript norm_features.R '+mtype+' '+infile+' '+meta+' '+dtype+' '+ofile)

def load_train_sp(infile,d,all_sp):
    f=open(infile,'r')
    samples=f.readline().strip().split()
    anno={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        c=0
        anno[ele[0]]=''
        for e in ele[1:]:
            d[ele[0]][samples[c]]=e
            c+=1
        all_sp[ele[0]]=''
    return samples,anno

def load_test_sp(infile,d,anno):
    f=open(infile,'r')
    samples=f.readline().strip().split()
    count=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        c=0
        if ele[0] in anno:
            count+=1
            for e in ele[1:]:
                d[ele[0]][samples[c]]=e
                c+=1
    #print(count,' species of training datasets are detected in test datasets.')
    return samples

def merge_sp(in1,in2,out):
    d=defaultdict(lambda:{})
    all_sp={}
    s1,anno=load_train_sp(in1,d,all_sp)
    s2=load_test_sp(in2,d,anno)
    samples=s1+s2
    o=open(out,'w+')
    o.write('\t'.join(samples)+'\n')
    for e in sorted(all_sp.keys()):
        o.write(e)
        for s in samples:
            if s in d[e]:
                o.write('\t'+str(d[e][s]))
            else:
                o.write('\t'+str(0))
        o.write('\n')

def load_item(infile,d,all_item):
    f=open(infile,'r')
    samples=f.readline().strip().split()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        c=0
        for e in ele[1:]:
            d[ele[0]][samples[c]]=e
            c+=1
        all_item[ele[0]]=''
    return samples
     
    
def merge_eggNOG(in1,in2,out):
    d=defaultdict(lambda:{})
    all_item={}
    s1=load_item(in1,d,all_item)
    s2=load_item(in2,d,all_item)
    samples=s1+s2
    o=open(out,'w+')
    o.write('\t'.join(samples)+'\n')
    for e in sorted(all_item.keys()):
        o.write(e)
        for s in samples:
            if s in d[e]:
                o.write('\t'+str(d[e][s]))
            else:
                o.write('\t'+str(0))
        o.write('\n')

def trans2node(infile,meta,ofile):
    f=open(meta,'r')
    status=[]
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t') 
        if ele[3]=='healthy':
            status.append('Health')
        else:
            status.append(ele[3])
    a=pd.read_table(infile)
    a=a.T
    a=np.array(a)
    c=0
    o=open(ofile,'w+')
    for t in a:
        o.write(str(c))
        for v in t:
            o.write('\t'+str(v))
        o.write('\t'+status[c]+'\n')
        c+=1
    o.close()
    
    
    

def main():
    usage="GDmicro_preprocess - Normalize all input data, merge your own test data with training data, and convert combined matrices to node feature format."
    parser=argparse.ArgumentParser(prog="GDmicro_preprcess.py",description=usage)
    parser.add_argument('-i','--input_train',dest='input_train',type=str,help="The dir of input training data.")
    parser.add_argument('-b','--input_test',dest='input_test',type=str,help="The dir of input test data.")
    parser.add_argument('-t','--train_mode',dest='train_mode',type=str,help="If set to 1, then will only normalize and convert all input data. This mode can only be used when input datasets are all training data. You don't need to provide the test data under this mode. (default: 0)")
    parser.add_argument('-d','--disease',dest='dtype',type=str,help="The name of disease. (Note: the value should be the same as the one in your metadata file.)")

    #parser.add_argument('-t','--data_type',dest='data_type',type=str,help="The type of input data. The value can be: meta (metadata), species (species matrix) or eggNOG  (eggNOG matrix). If set to \"species\" or \"eggNOG\", you also need to set \"-m\" and \"-n\" to provide the metadata of training and test datasets..")
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of combined and normalized results. (Default: GDmicro_merge")
    args=parser.parse_args()

    intrain=args.input_train
    intest=args.input_test
    train_mode=args.train_mode
    dtype=args.dtype
    out=args.outdir

    if not out:
        out="GDmicro_merge"
    
    if not os.path.exists(out):
        os.makedirs(out)

    if not train_mode:
        train_mode=0
    else:
        train_mode=int(train_mode)

    
    intrain_meta=''
    intest_meta=''
    #intest_meta_nl=''
    intrain_sp=''
    intest_sp=''
    intrain_eggnog=''
    intest_eggnog=''
    for filename in os.listdir(intrain):
        if re.search('meta\.tsv',filename):
            intrain_meta=intrain+'/'+filename
        if re.search('sp_matrix',filename):
            intrain_sp=intrain+'/'+filename
        if re.search('eggNOG_matrix',filename):
            intrain_eggnog=intrain+'/'+filename
    if train_mode==0:
        for filename in os.listdir(intest):
            if re.search('meta\.tsv',filename):
                intest_meta=intest+'/'+filename
            if re.search('sp_matrix',filename):
                intest_sp=intest+'/'+filename
            if re.search('eggNOG_matrix',filename):
                intest_eggnog=intest+'/'+filename
    if train_mode==0:
        check_arr=[intrain_meta,intest_meta,intrain_sp,intest_sp,intrain_eggnog,intest_eggnog]
    else:
        check_arr=[intrain_meta,intrain_sp,intrain_eggnog]
    # 
    for i in check_arr:
        if i=='':
            print('Some required files are not provided. Please check!')
            exit()
        else:
            print("Load files -> "+i)
    #exit()
    if train_mode==0:
        print('Preprocess 1 - Merge metadata.')
        trans_meta(intrain_meta,intest_meta,out+"/"+dtype+'_meta.tsv')
        print('Preprocess 2 - Normalize all abundance matrices.')
        normalize_data(intrain_sp,'species',intrain_meta,dtype,out+"/"+dtype+'_train_sp_norm.csv')
        normalize_data(intest_sp,'species',intest_meta,dtype,out+"/"+dtype+'_test_sp_norm.csv')
        normalize_data(intrain_eggnog,'eggNOG',intrain_meta,dtype,out+"/"+dtype+'_train_eggNOG_norm.csv')
        normalize_data(intest_eggnog,'eggNOG',intest_meta,dtype,out+"/"+dtype+'_test_eggNOG_norm.csv')
    
        print('Preprocess 3 - Merge training and test datasets.') 
        merge_sp(intrain_sp,intest_sp,out+"/"+dtype+'_sp_merge_raw.csv')
        merge_sp(out+"/"+dtype+'_train_sp_norm.csv',out+"/"+dtype+'_test_sp_norm.csv',out+"/"+dtype+'_sp_merge_norm.csv')
        merge_eggNOG(intrain_eggnog,intest_eggnog,out+"/"+dtype+'_eggNOG_merge_raw.csv')
        merge_eggNOG(out+"/"+dtype+'_train_eggNOG_norm.csv',out+"/"+dtype+'_test_eggNOG_norm.csv',out+"/"+dtype+'_eggNOG_merge_norm.csv')
        print('Preprocess 4 - Convert combined matrices to node feature format.')
        trans2node(out+"/"+dtype+'_sp_merge_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_merge_norm_node.csv')
        #trans2node(out+"/"+dtype+'_eggNOG_merge_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_eggNOG_merge_norm_node.csv')
    else:
        print('Train mode - Preprocess 1 - Transform metadata.')
        trans_meta_train(intrain_meta,out+"/"+dtype+'_meta.tsv')
        print('Train mode - Preprocess 2 - Normalize all abundance matrices.')
        normalize_data(intrain_sp,'species',intrain_meta,dtype,out+"/"+dtype+'_train_sp_norm.csv')
        os.system('cp '+intrain_eggnog+' '+out+"/"+dtype+'_train_eggNOG_raw.csv')
        normalize_data(intrain_eggnog,'eggNOG',intrain_meta,dtype,out+"/"+dtype+'_train_eggNOG_norm.csv')
        print('Train mode - Preprocess 3 - Convert normalized matrices to node feature format.')
        trans2node(out+"/"+dtype+'_train_sp_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_train_norm_node.csv')

    if os.path.exists(intrain+'/pre_features'):
        os.system('cp -rf '+intrain+'/pre_features '+out)

        

if __name__=="__main__":
    sys.exit(main())







