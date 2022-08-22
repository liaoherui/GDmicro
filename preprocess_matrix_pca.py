import re
import os
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt

def load_ph(f):
    d={}
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        if ele[3]=='healthy':
            d['S'+ele[0]]='Health'
        else:
            d['S'+ele[0]]=ele[3]
    return d


def preprocess(infile,phenotype):
    f1=open(infile,'r')
    f2=open(phenotype,'r')
    d=load_ph(f2)
    X=[]
    y=[]
    samples=[]
    while True:
        line=f1.readline().strip()
        if not line:break
        ele=re.split(',',line)
        y.append(d[ele[0]])
        samples.append(ele[0])
        tem=[]
        for e in ele[1:]:
            tem.append(float(e))
        X.append(tem)
    X=np.array(X)

    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    
    #X=preprocessing.scale(X)
    #print(X)
    #exit()
    return X,y,samples


def pca(X,y,samples,outfig,ptitle,omatrix):
    #print(X,y)
    pca=PCA(n_components=0.95)
    reduced_x=pca.fit_transform(X)
    crc_x,crc_y=[],[]
    health_x,health_y=[],[]
    #print(len(X),len(reduced_x),len(y),y)
    dname='' 
    for i in range(len(reduced_x)):
        #print(len(reduced_x[i]))
        if not y[i]=='Health':
            crc_x.append(reduced_x[i][0])
            crc_y.append(reduced_x[i][1])
            dname=y[i]
        else:
            health_x.append(reduced_x[i][0])
            health_y.append(reduced_x[i][1])
    '''
    plt.figure()
    plt.scatter(crc_x,crc_y,c='r',marker='x',label=dname)
    plt.scatter(health_x,health_y,c='g',marker='D',label='Health')
    plt.legend()
    plt.title(ptitle)
    plt.savefig(outfig,dpi=400)
    '''
    o=open(omatrix,'w+')
    i=0
    for s in samples:
        o.write(s)
        for e in reduced_x[i]:
            o.write(','+str(e))
        o.write('\n')
        i+=1
    o.close()

def run_pca(check1,check2,inmatrix,metadata,pre,out):
    if os.path.exists(check1) or os.path.exists(check2):
        X,y,samples=preprocess(inmatrix,metadata)
        pca(X,y,samples,out+'/'+pre+'_pca_res.png',pre+'_extracted_features',out+'/'+pre+'_matrix_ef_pca.csv')

#run_pca('species_associate.pdf','species_auc_run.txt','species_embedding_vector.txt','../../Graph_with_raw_data_from_paper_Merge/EMG_LOO_test_AUS_109/sample_phenotype.txt','species','train_embedding')


