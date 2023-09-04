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

def extract_tout(tout,d,ofile):
    o=open(ofile,'w+')
    f=open(tout,'r')
    line=f.readline().strip()
    ele=line.split('\t')
    dc={}
    c=0
    arr=[]
    for e in ele:
        if e in d:
            dc[c]=''
            arr.append(e)
        c+=1
    o.write('\t'.join(arr)+'\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        o.write(ele[0])
        c=0
        tem=[]
        for e in ele[1:]:
            if c in dc:
                #o.write('\t'+e)
                tem.append(e)
            c+=1
        o.write('\t'+'\t'.join(tem)+'\n')
    o.close()


def normalize_data_small(infile,mtype,meta,dtype,ofile,inmerge,inmerge_meta):
    f=open(meta,'r')
    meta_content=[]
    line=f.readline().strip()
    meta_content.append(line)
    c=0
    ag=0
    d={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        d[ele[2]]=''
        meta_content.append(line)
        label=line.split('\t')[3]
        if label=='Unknown':
            ag=1
        c+=1
    uid = uuid.uuid1().hex
    if ag==1:
        n_split_d=int(c/2)
        n_split_h=c-n_split_d
        ml_d=[dtype for i in range(n_split_d)]
        ml_h=['healthy' for i in range(n_split_h)]
        ml=ml_d+ml_h
        #print(ml)
        #exit()
        if len(meta_content[1:])<13:
            tmeta = 'tem_meta_' + uid + '.tsv'
            ot = open(tmeta, 'w+')
            ot.write(meta_content[0]+'\n')
            ft=open(inmerge_meta,'r')
            line=ft.readline()
            i=0
            while True:
                line=ft.readline().strip()
                if not line:break
                ele=line.split('\t')
                if ele[-1]=='train':
                    ot.write('\t'.join(ele[:-1])+'\n')
                else:
                    ele[3]=ml[i]
                    i+=1
                    ot.write('\t'.join(ele[:-1])+'\n')
            ot.close()
            tout='tem_matrix_' + uid + '.tsv'

            os.system('Rscript norm_features.R ' + mtype + ' ' + inmerge + ' ' + tmeta + ' ' + dtype + ' ' + tout)
            extract_tout(tout,d,ofile)
            os.system('rm ' + tmeta+' '+tout)
        else:
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
        if len(meta_content[1:]) < 13:
            tout = 'tem_matrix_' + uid + '.tsv'
            os.system('Rscript norm_features.R ' + mtype + ' ' + inmerge + ' ' + inmerge_meta + ' ' + dtype + ' ' + tout)
            extract_tout(tout, d, ofile)
            os.system('rm ' + tout)
        else:
            os.system('Rscript norm_features.R '+mtype+' '+infile+' '+meta+' '+dtype+' '+ofile)

def normalize_data(infile,mtype,meta,dtype,ofile):
    f=open(meta,'r')
    meta_content=[]
    line=f.readline().strip()
    meta_content.append(line)
    c=0
    ag=0
    #d={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        #d[ele[2]]=''
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
        uid = uuid.uuid1().hex

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

def split_file(infile,disease,outdir):
    intrain=outdir+'/Split_dir/Train'
    intest=outdir+'/Split_dir/Test'
    if not os.path.exists(intrain):
        os.makedirs(intrain)
    if not os.path.exists(intest):
        os.makedirs(intest)

    f=open(infile,'r')
    line=f.readline().strip()
    ele=re.split(',',line)
    sp=ele[4:]
    train_ab=[]
    test_ab=[]
    sample_train=[]
    sample_test=[]
    train_meta={}
    test_meta={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        if ele[1]=='train':
            sample_train.append(ele[0])
            train_meta[ele[0]]=[ele[2],ele[3]]
            train_ab.append(ele[4:])
        else:
            sample_test.append(ele[0])
            test_meta[ele[0]]=[ele[2],ele[3]]
            test_ab.append(ele[4:])
    o1=open(intrain+'/'+disease+'_meta.tsv','w+')
    o2=open(intrain+'/'+disease+'_sp_matrix.csv','w+')
    o1.write('sampleID\tstudyName\tsubjectID\tdisease\tcountry\n')
    c=0
    for s in sample_train:
        o1.write(str(c)+'\t'+train_meta[s][1]+'\t'+s+'\t'+train_meta[s][0]+'\t'+train_meta[s][1]+'\n')
        c+=1
    o2.write('\t'.join(sample_train)+'\n')
    c=0
    train_ab=np.array(train_ab)
    train_ab=train_ab.T
    for s in sp:
        tab=0
        for x in train_ab[c]:
            tab+=float(x)
        if tab==0:
            c+=1
            continue
        o2.write(s+'\t'+'\t'.join(train_ab[c])+'\n') 
        c+=1
    if len(sample_test)>0:
        o3=open(intest+'/'+disease+'_meta.tsv','w+')
        o4=open(intest+'/'+disease+'_sp_matrix.csv','w+')
        o3.write('sampleID\tstudyName\tsubjectID\tdisease\tcountry\n')
        c=0
        for s in sample_test:
            o3.write(str(c)+'\t'+test_meta[s][1]+'\t'+s+'\t'+test_meta[s][0]+'\t'+test_meta[s][1]+'\n')
            c+=1
        o4.write('\t'.join(sample_test)+'\n')
        c=0
        test_ab=np.array(test_ab)
        test_ab=test_ab.T
        for s in sp:
            tab=0
            for x in test_ab[c]:
                tab+=float(x)
            if tab==0:
                c+=1
                continue
            o4.write(s+'\t'+'\t'.join(test_ab[c])+'\n')
            c+=1
    return intrain,intest

def check_test_num(meta):
    f=open(meta,'r')
    line=f.readline()
    c=0
    while True:
        line=f.readline()
        if not line:break
        c+=1
    if c<13:
        return True
    else:
        return False


def scan_test_num(infile):
    f=open(infile,'r')
    arr=[]
    tn=0
    line=f.readline().strip()
    arr.append(line)
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        if ele[1]=='test':
            ele[2]='Unknown'
            tn+=1
        tem=','.join(ele)
        arr.append(tem)
    if tn<13:
        uid = uuid.uuid1().hex
        ninfile='inmatrix_'+uid+'.csv'
        o=open(ninfile,'w+')
        for a in arr:
            o.write(a+'\n')
        o.close()
        return ninfile
    else:
        return ''

    
    

def preprocess(infile,train_mode,disease,outdir):
    '''
    usage="GDmicro_preprocess - Normalize all input data, merge your own test data with training data, and convert combined matrices to node feature format."
    parser=argparse.ArgumentParser(prog="GDmicro_preprcess.py",description=usage)
    parser.add_argument('-i','--input_train',dest='input_train',type=str,help="The dir of input training data.")
    parser.add_argument('-b','--input_test',dest='input_test',type=str,help="The dir of input test data.")
    parser.add_argument('-t','--train_mode',dest='train_mode',type=str,help="If set to 1, then will only normalize and convert all input data. This mode can only be used when input datasets are all training data. You don't need to provide the test data under this mode. (default: 0)")
    parser.add_argument('-d','--disease',dest='dtype',type=str,help="The name of disease. (Note: the value should be the same as the one in your metadata file.)")

    #parser.add_argument('-t','--data_type',dest='data_type',type=str,help="The type of input data. The value can be: meta (metadata), species (species matrix) or eggNOG  (eggNOG matrix). If set to \"species\" or \"eggNOG\", you also need to set \"-m\" and \"-n\" to provide the metadata of training and test datasets..")
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of combined and normalized results. (Default: GDmicro_merge")
    args=parser.parse_args()
    '''

    #intrain=args.input_train
    #intest=args.input_test
    scan_res=scan_test_num(infile)
    if not scan_res=='':
        infile=scan_res
    #print(infile)
    #exit()
    intrain,intest=split_file(infile,disease,outdir)

    train_mode=train_mode
    dtype=disease
    out=outdir+'/Preprocess_data'
    '''
    if not out:
        out="GDmicro_merge"
    '''
    
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
        '''
        if re.search('eggNOG_matrix',filename):
            intrain_eggnog=intrain+'/'+filename
        '''
    if train_mode==0:
        for filename in os.listdir(intest):
            if re.search('meta\.tsv',filename):
                intest_meta=intest+'/'+filename
            if re.search('sp_matrix',filename):
                intest_sp=intest+'/'+filename
            '''
            if re.search('eggNOG_matrix',filename):
                intest_eggnog=intest+'/'+filename
            '''
    if train_mode==0:
        check_arr=[intrain_meta,intest_meta,intrain_sp,intest_sp]
    else:
        check_arr=[intrain_meta,intrain_sp]
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
        #exit()
        if check_test_num(intest_meta):
            merge_sp(intrain_sp,intest_sp,outdir+"/Split_dir/Test/"+dtype+'_merge_sp_raw.csv')
            temerge=outdir+"/Split_dir/Test/"+dtype+'_merge_sp_raw.csv'
            os.system('cp '+out+"/"+dtype+'_meta.tsv'+' '+outdir+"/Split_dir/Test/"+dtype+'_meta_merge.tsv')
            temeta=outdir+"/Split_dir/Test/"+dtype+'_meta_merge.tsv'
        print('Preprocess 2 - Normalize all abundance matrices.')
        normalize_data(intrain_sp,'species',intrain_meta,dtype,out+"/"+dtype+'_train_sp_norm.csv')
        if check_test_num(intest_meta):
            normalize_data_small(intest_sp, 'species', intest_meta, dtype, out + "/" + dtype + '_test_sp_norm.csv',temerge,temeta)
            #exit()
        else:
            normalize_data(intest_sp,'species',intest_meta,dtype,out+"/"+dtype+'_test_sp_norm.csv')
        #normalize_data(intrain_eggnog,'eggNOG',intrain_meta,dtype,out+"/"+dtype+'_train_eggNOG_norm.csv')
        #normalize_data(intest_eggnog,'eggNOG',intest_meta,dtype,out+"/"+dtype+'_test_eggNOG_norm.csv')
    
        print('Preprocess 3 - Merge training and test datasets.') 
        merge_sp(intrain_sp,intest_sp,out+"/"+dtype+'_sp_merge_raw.csv')
        merge_sp(out+"/"+dtype+'_train_sp_norm.csv',out+"/"+dtype+'_test_sp_norm.csv',out+"/"+dtype+'_sp_merge_norm.csv')
        #merge_eggNOG(intrain_eggnog,intest_eggnog,out+"/"+dtype+'_eggNOG_merge_raw.csv')
        #merge_eggNOG(out+"/"+dtype+'_train_eggNOG_norm.csv',out+"/"+dtype+'_test_eggNOG_norm.csv',out+"/"+dtype+'_eggNOG_merge_norm.csv')
        print('Preprocess 4 - Convert combined matrices to node feature format.')
        trans2node(out+"/"+dtype+'_sp_merge_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_merge_norm_node.csv')
        trans2node(out+"/"+dtype+'_sp_merge_raw.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_merge_raw_node.csv')
        #trans2node(out+"/"+dtype+'_eggNOG_merge_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_eggNOG_merge_norm_node.csv')
    else:
        print('Train mode - Preprocess 1 - Transform metadata.')
        trans_meta_train(intrain_meta,out+"/"+dtype+'_meta.tsv')
        print('Train mode - Preprocess 2 - Normalize all abundance matrices.')
        os.system('cp '+intrain_sp+' '+out+"/"+dtype+'_train_sp_raw.csv')
        normalize_data(intrain_sp,'species',intrain_meta,dtype,out+"/"+dtype+'_train_sp_norm.csv')
        #os.system('cp '+intrain_eggnog+' '+out+"/"+dtype+'_train_eggNOG_raw.csv')
        #normalize_data(intrain_eggnog,'eggNOG',intrain_meta,dtype,out+"/"+dtype+'_train_eggNOG_norm.csv')
        print('Train mode - Preprocess 3 - Convert normalized matrices to node feature format.')
        trans2node(out+"/"+dtype+'_train_sp_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_train_norm_node.csv')
        trans2node(out+"/"+dtype+'_train_sp_raw.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_train_raw_node.csv')
    '''
    if os.path.exists(intrain+'/pre_features'):
        os.system('cp -rf '+intrain+'/pre_features '+out)
    '''
    #exit()
    if not scan_res=='':
        os.system('rm '+scan_res)
    return out

        
'''
if __name__=="__main__":
    sys.exit(main())
'''


#preprocess('../Single_test_datasets/train_5studies_1test_single/GuptaA_2019_test_1.csv',0,'CRC','single_sample')




