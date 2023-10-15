import re
import os
import numpy as np
import higra as hg
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def check_trans_visualize_graph(sinfo,outgraph,out,pre,olog):
    G=nx.Graph()
    f=open(sinfo,'r')
    d={}
    line=f.readline()
    all_case=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        if not ele[3]=='healthy':
            d['S'+ele[0]]=ele[3]
        else:
            d['S'+ele[0]]='Health'
        all_case.append(ele[3])
    all_edges=[]

    disease=[]
    health=[]
    #print(outgraph)
    f22=open(outgraph,'r')
    o=open(out+'/'+pre+'_pca_knn_graph_final.txt','w+')
    while True:
        line=f22.readline().strip()
        if not line:break
        #print(line)
        ele=line.split()
        o.write(re.sub('S','',ele[0])+'\t'+re.sub('S','',ele[1])+'\n')
        edge=(ele[0],ele[1])
        all_edges.append(edge)
        if not d[ele[0]]=='Health':
            if ele[0] not in disease:disease.append(ele[0])
        else:
            if ele[0] not in health:health.append(ele[0])
        if not d[ele[1]]=='Health':
            if ele[1] not in disease:disease.append(ele[1])
        else:
            if ele[1] not in health:health.append(ele[1])

    #print(all_edges)
    #exit()
    o.close()

    G.add_edges_from(all_edges)
    print('The number of edges of '+pre+' PCA KNN graph:',G.number_of_edges())
    olog.write('The number of edges of '+pre+' PCA KNN graph: '+str(G.number_of_edges())+'\n')
    print('Whether '+pre+' PCA KNN graph connected? ',nx.is_connected(G),'\n')
    olog.write('Whether '+pre+' PCA KNN graph connected? '+str(nx.is_connected(G))+'\n\n')
    pos=nx.spring_layout(G,seed=3113794652)
    plt.figure()
    color_map=[]
    for node in G:
        if node in disease:
            color_map.append('red')
        else:
            color_map.append('green')
    nx.draw(G,node_size=400,node_color=color_map,with_labels = True,font_size=8)
    
    for i in set(all_case):
        if i=='Health':
            plt.scatter([],[], c=['green'], label='{}'.format(i))
        else:
            plt.scatter([],[], c=['red'], label='{}'.format(i))
    plt.legend()
    plt.savefig(out+'/'+pre+'_pca_knn_graph_final.png',dpi=400)
    



def construct_g(check1,check2,imatrix,sinfo,knn_nn,out,pre,olog,rfile):
    r=0
    if not os.path.exists(check1):
        r+=1
    if not os.path.exists(check2) :
        r+=1
    if r==2:
        return
    f1=open(sinfo,'r')
    d={} # Sample -> label
    line=f1.readline().strip()
    dname=''
    drname={}
    while True:
        line=f1.readline().strip()
        if not line:break
        ele=line.split()
        drname['S'+ele[0]]=ele[2]
        if not ele[3]=='healthy':
            d['S'+ele[0]]=ele[3]
            dname=ele[3]
        else:
            d['S'+ele[0]]='Health'
    f2=open(imatrix,'r')
    X=[]
    y=[]
    did2name={}
    count=0
    while True:
        line=f2.readline().strip()
        if not line:break
        ele=re.split(',',line)
        y.append(d[ele[0]])
        tmp=[]
        for e in ele[1:]:
            tmp.append(float(e))
        X.append(tmp)
        did2name[count]=ele[0]
        count+=1
    X=np.array(X)
    graph,edge_weights=hg.make_graph_from_points(X, graph_type='knn',n_neighbors=knn_nn)
    sources, targets = graph.edge_list()
    #print(sources)
    #exit()
    
    outgraph=out+'/'+pre+'_pca_knn_graph_ini.txt'

    drecord=defaultdict(lambda:{})
    o=open(outgraph,'w+')
    for i in range(len(sources)):
        o.write(did2name[sources[i]]+'\t'+did2name[targets[i]]+'\t'+str(edge_weights[i])+'\n')
        drecord[did2name[sources[i]]][did2name[targets[i]]]=str(edge_weights[i])
        drecord[did2name[targets[i]]][did2name[sources[i]]]=str(edge_weights[i])
    o.close()
    #o.close()
    #exit()
    correct=0
    total=len(X)
    ot=open(rfile,'w+')
    ot.write('All_samples\tNeighbors\n')
    for r in drecord:
        cl=d[r]
        dn=0
        hn=0
        fl=''
        for e in drecord[r]:
            if d[e]=='Health':
                hn+=1
            else:
                dn+=1
        if hn>dn:
            fl='Health'
        if dn>hn:
            fl=dname
        if cl==fl:
            correct+=1
        ot.write(drname[r]+'\t')
        tem=[]
        for e in drecord[r]:
            tem.append(drname[e]+':'+d[e]+':'+drecord[r][e])
        ot.write('\t'.join(tem)+'\n')
        #print(r,cl,fl,hn,dn)
    print('The acc of '+pre+' knn graph: ',correct/total,correct,'/',total)
    olog.write('The acc of '+pre+' knn graph: '+str(float(correct/total))+' '+str(correct)+'/'+str(total)+'\n')
    #exit()
    check_trans_visualize_graph(sinfo,outgraph,out,pre,olog) 
    

