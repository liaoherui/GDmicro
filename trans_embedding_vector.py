import re
import os

def trans(infile,insample,ofile):
    fp=open(insample,'r')
    line=fp.readline().strip()
    samples=[]
    while True:
        line=fp.readline().strip()
        if not line:break
        ele=line.split('\t')
        samples.append('S'+str(ele[0]))

    f=open(infile,'r')
    #line=f.readline()
    o=open(ofile,'w+')
    c=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        te=[]
        for e in ele:
            te.append(str(float(e)))
        o.write(samples[c]+','+','.join(te)+'\n')
        c+=1
#trans('../feature_out.txt','../../Graph_with_raw_data_from_paper_Merge/EMG_LOO_test_AUS_109/sample_phenotype.txt','species_embedding_vector.txt')
