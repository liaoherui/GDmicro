import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
#import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
#########################
from sklearn.model_selection import StratifiedKFold
import random
#import calculate_avg_acc_of_cross_validation_test
from sklearn import metrics
from scipy import stats



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden = 32
dropout = 0.5
lr = 0.01 
weight_decay = 1e-5
fastmode = 'store_true'


def encode_onehot(labels):
    #classes=set(labels)
    classes_old=sorted(list(set(labels)),reverse=True)
    classes=[]
    s=0
    for c in classes_old:
        if c=='Unknown':
            s=1
            continue
        if c=='Health':continue
        classes.append(c)
    if s==1:
        classes.append('Health')
        classes.append('Unknown')
    else:
        classes.append('Health')

    classes_dict={c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot,classes_dict

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(mlp_or_not,graph,node_file,input_sample):
    ##### Load input sample info ######
    f=open(input_sample,'r')
    line=f.readline()
    train_id={}
    idx_train=[]
    idx_test=[]
    lid=0
    c=0
    tid2name={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()    
        if ele[-1]=='train':
            train_id['S'+ele[1]]=''
            if int(ele[1])>lid:
                lid=int(ele[1])
            idx_train.append(c)
        else:
            idx_test.append(c)
        t2d2name[c].append(ele[2])
        c+=1





    print('Loading {} dataset...'.format(graph+' plus '+node_file))
    idx_features_labels = np.genfromtxt("{}".format(node_file),dtype=np.dtype(str))
    #print(idx_features_labels)
    features=idx_features_labels[:, 1:-1]
    features=features.astype(float)
    
    a=np.array(idx_features_labels[:, 1:-1])
    a=a.astype(float)
    #a=stats.zscore(a,axis=1,ddof=1)
    '''
    features=[]
    for s in a:
        mean=s.mean()
        std=s.std()
        features.append((s-mean)/std)
    '''
    features=np.array(features)
    #print(features)
    #exit()
    #features=a
    
    #print(features)
    #exit()
    
    #features = sp.csr_matrix(features, dtype=np.float32)
    #features = normalize(features)
    #features = torch.FloatTensor(np.array(features.todense()))
    #print(idx_features_labels[:, -1])
    #exit()
    labels,classes_dict = encode_onehot(idx_features_labels[:, -1])
    #features_train
    #labels_train
    #print(idx_features_labels[:, -1])
    #print(len(labels))
    #labels = torch.LongTensor(np.where(labels)[1])
    f1=features[idx_train]
    f2=features[idx_test]
    l1=labels[idx_train]
    l2=labels[idx_test]
    features=np.concatenate((f1, f2), axis=0)
    labels=np.concatenate((l1, l2), axis=0)
    #print(features)
    #print(labels)
    #exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}".format(graph),dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #### identity matrix
    if mlp_or_not=='mlp':
        adj=sp.identity(len(labels)).toarray()
        adj=sp.csr_matrix(adj)
    else:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    #print(adj.shape)
    #print(adj)
    #exit()
    
    total_num=len(labels)
    #tnum=int(3*(total_num/4))

    #idx_train = range(tnum)
    #idx_val = range(tnum,total_num)
    #print(idx_train)
    #print(idx_val)
    #exit()
    idx_test = range(len(idx_train), len(labels))
    #print(len(idx_train))
    #print(idx_test)
    #exit()

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    features_train=features[:len(idx_train)]
    labels_train=labels[:len(idx_train)]

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #print(labels)
    return adj, features, labels, features_train,labels_train, idx_test,idx_train,classes_dict,tid2name

#adj, features, labels, idx_train, idx_val, idx_test=load_data()
#splits=KFold(n_splits=10,shuffle=True,random_state=1234)

class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
        else:
            lf.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self,input,adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' +str(self.in_features) + ' -> '+str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
    '''
    def forward(self,x,adj):
        h1=F.relu(self.gc1(x, adj))
        logits = self.gc2(h1, adj)
        return logits
    '''
    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)

#model = GCN(nfeat=features.shape[1], nhid=hidden,nclass=labels.max().item() + 1,dropout=dropout)

def accuracy(output,labels):
    preds=output.max(1)[1].type_as(labels)
    correct=preds.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)

def AUC(output,labels):
    #print(output.data.numpy())
    output=torch.exp(output)
    a=output.data.numpy()
    preds=a[:,1]
    #exit()
    #preds=output.max(1)[0].data.numpy()
    #print(preds,output.max(1)[1])
    #print(labels)
    #exit()
    #preds=output.max(1)[1].type_as(labels)
    #print(np.array(preds),np.array(labels))
    #exit()
    fpr,tpr,thresholds=metrics.roc_curve(np.array(labels),np.array(preds))
    auc=metrics.auc(fpr,tpr)
    #print(fpr,tpr)
    #exit()
    return auc

def train(epoch,idx_train_in,idx_val_in,model,optimizer,features,adj,labels,o,max_val_auc,rdir,fold,classes_dict,tid2name,wwl,record,close_cv):
    #model.to(device).super().reset_parameters()
    #model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout)
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    t=time.time()
    model.train()
    optimizer.zero_grad()
    output=model(features,adj)
    loss_train=torch.nn.functional.nll_loss(output[idx_train_in], labels[idx_train_in])
    acc_train = accuracy(output[idx_train_in], labels[idx_train_in])
    auc_train=AUC(output[idx_train_in], labels[idx_train_in])
    loss_train.backward()
    optimizer.step()

    #if not fastmode:
    model.eval()
    output=model(features,adj)
    #loss_val = torch.nn.functional.nll_loss(output[idx_val_in], labels[idx_val_in])
    if close_cv==0:
            loss_val = torch.nn.functional.nll_loss(output[idx_val_in], labels[idx_val_in])
            acc_val = accuracy(output[idx_val_in], labels[idx_val_in])
            auc_val = AUC(output[idx_val_in], labels[idx_val_in])
    if close_cv==0:
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val.item()),'time: {:.4f}s'.format(time.time() - t),'AUC_train: {:.4f}'.format(auc_train.item()),'AUC_val: {:.4f}'.format(auc_val.item()))

        if wwl==1:
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' loss_val: {:.4f}'.format(loss_val.item())+' acc_val: {:.4f}'.format(acc_val.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+' AUC_val: {:.4f}'.format(auc_val.item())+'')
        else:
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' loss_val: {:.4f}'.format(loss_val.item())+' acc_val: {:.4f}'.format(acc_val.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+' AUC_val: {:.4f}'.format(auc_val.item())+'\n')
        if auc_val>max_val_auc and record==1:
            o3=open(rdir+'/sample_prob_fold'+str(fold)+'_val.txt','w+')
            output_res=torch.exp(output[idx_val_in])
            output_res=output_res.data.numpy()
            c=0
            dt={}
            for n in classes_dict:
                if n=="Unknown":continue
                if int(classes_dict[n][0])==1:
                    dt[0]=n
                else:
                    dt[1]=n
            for a in output_res:
                nt=labels[idx_val_in[c]].data.numpy()
                o3.write(tid2name[int(idx_val_in[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_val_in[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
                c+=1
        return auc_val
    else:
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()))
        if wwl==1:
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' AUC_train: {:.4f}'.format(auc_train.item())+'')
        else:
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' AUC_train: {:.4f}'.format(auc_train.item())+'\n')
        return auc_train

def train_fs(epoch,idx_train_in,idx_val_in,model,optimizer,features,adj,labels,o,max_val_auc,rdir,fold,classes_dict,tid2name,wwl,record,close_cv):
    #model.to(device).super().reset_parameters()
    #model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout)
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    t=time.time()
    model.train()
    optimizer.zero_grad()
    output=model(features,adj)
    loss_train=torch.nn.functional.nll_loss(output[idx_train_in], labels[idx_train_in])
    acc_train = accuracy(output[idx_train_in], labels[idx_train_in])
    auc_train=AUC(output[idx_train_in], labels[idx_train_in])
    loss_train.backward()
    optimizer.step()

    #if not fastmode:
    model.eval()
    output=model(features,adj)
    #loss_val = torch.nn.functional.nll_loss(output[idx_val_in], labels[idx_val_in])
    auc_val=0
    if close_cv==0 and wwl==1:
            loss_val = torch.nn.functional.nll_loss(output[idx_val_in], labels[idx_val_in])
            acc_val = accuracy(output[idx_val_in], labels[idx_val_in])
            auc_val = AUC(output[idx_val_in], labels[idx_val_in])
    if close_cv==0:       
        if wwl==1:
            print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val.item()),'time: {:.4f}s'.format(time.time() - t),'AUC_train: {:.4f}'.format(auc_train.item()),'AUC_val: {:.4f}'.format(auc_val.item()))
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' loss_val: {:.4f}'.format(loss_val.item())+' acc_val: {:.4f}'.format(acc_val.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+' AUC_val: {:.4f}'.format(auc_val.item())+'')
        else:
            print('Epoch: {:04d}'.format(epoch+1),' loss_train: {:.4f}'.format(loss_train.item()),' acc_train: {:.4f}'.format(acc_train.item()),' time: {:.4f}s'.format(time.time() - t),' AUC_train: {:.4f}'.format(auc_train.item()))
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+'\n')
        if auc_val>max_val_auc and record==1:
            o3=open(rdir+'/sample_prob_fold'+str(fold)+'_val.txt','w+')
            output_res=torch.exp(output[idx_val_in])
            output_res=output_res.data.numpy()
            c=0
            dt={}
            for n in classes_dict:
                if n=="Unknown":continue
                if int(classes_dict[n][0])==1:
                    dt[0]=n
                else:
                    dt[1]=n
            for a in output_res:
                nt=labels[idx_val_in[c]].data.numpy()
                o3.write(tid2name[int(idx_val_in[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_val_in[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
                c+=1
        return auc_train,torch.exp(output).data.numpy()
    else:
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()))
        if wwl==1:
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' AUC_train: {:.4f}'.format(auc_train.item())+'')
        else:
            o.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' AUC_train: {:.4f}'.format(auc_train.item())+'\n')
        return auc_train,torch.exp(output).data.numpy()
        #if auc_train>max_val_auc and record==1:


        



'''
def test():
    model.eval()
    output=model(features,adj)
    loss_test = torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    print(preds,labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:","loss= {:.4f}".format(loss_test.item()),"accuracy= {:.4f}".format(acc_test.item()))

def test_pred():
    model.eval()
    output=model(features,adj)
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    print(preds)
'''

#optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

def test_unknown(model,idx_test,features,adj,rdir,fn,classes_dict,tid2name,record):
    model.eval()
    output=model(features,adj)
    #loss_test=torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    #preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    if record==1:
        o3=open(rdir+'/sample_prob_fold'+str(fn)+'_test.txt','w+')
        output_res=torch.exp(output[idx_test])
        output_res=output_res.data.numpy()
        c=0
        dt={}
        for n in classes_dict:
            if n=="Unknown":continue
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            if a[0]>a[1]:
                res=0
            else:
                res=1
            o3.write(tid2name[int(idx_test[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(res)+'\t'+str(dt[res])+'\n')
            c+=1



def test(model,idx_test,features,adj,labels,o,max_test_auc,rdir,fn,classes_dict,tid2name,record):
    model.eval()
    output=model(features,adj)
    loss_test=torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    #print(preds,labels[idx_test])
    #exit()
    acc_test=accuracy(output[idx_test],labels[idx_test])
    auc_test=AUC(output[idx_test], labels[idx_test])
    print(" | Test set results:","loss={:.4f}".format(loss_test.item()),"accuracy={:.4f}".format(acc_test.item()),"AUC={:.4f}".format(auc_test.item()))
    o.write(" | Test set results:"+"loss={:.4f}".format(loss_test.item())+" accuracy: {:.4f}".format(acc_test.item())+" AUC: {:.4f}".format(auc_test.item())+'\n')
    if auc_test>max_test_auc and record==1:
        o3=open(rdir+'/sample_prob_fold'+str(fn)+'_test.txt','w+')
        output_res=torch.exp(output[idx_test])
        output_res=output_res.data.numpy()
        c=0
        dt={}
        for n in classes_dict:
            if n=="Unknown":continue
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            nt=labels[idx_test[c]].data.numpy()
            o3.write(tid2name[int(idx_test[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_test[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
            c+=1
    return auc_test

def test_new_acc(model,idx_test,features,adj,labels,o,max_test_acc,rdir,fn,classes_dict,tid2name,record):
    model.eval()
    output=model(features,adj)
    loss_test=torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    acc_test=accuracy(output[idx_test],labels[idx_test])
    auc_test=AUC(output[idx_test], labels[idx_test])
    print(" | Test set results:","loss={:.4f}".format(loss_test.item()),"accuracy={:.4f}".format(acc_test.item()),"AUC={:.4f}".format(auc_test.item()))
    o.write(" | Test set results:"+"loss={:.4f}".format(loss_test.item())+" accuracy: {:.4f}".format(acc_test.item())+" AUC: {:.4f}".format(auc_test.item())+'\n')
    if acc_test>max_test_acc and record==1:
        o3=open(rdir+'/sample_prob_fold'+str(fn)+'_test.txt','w+')
        output_res=torch.exp(output[idx_test])
        output_res=output_res.data.numpy()
        c=0
        dt={}
        for n in classes_dict:
            if n=="Unknown":continue
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            nt=labels[idx_test[c]].data.numpy()
            o3.write(tid2name[int(idx_test[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_test[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
            c+=1
    return acc_test

def run_GCN_test(mlp_or_not,epochs,graph,node_file,outfile1,outfile2,input_sample):
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 16
    lr = 0.01
    weight_decay = 5e-4
    fastmode = 'store_true'
    '''

    adj,features,labels,features_train,labels_train,idx_test,idx_train,classes_dict,tid2name=load_data(mlp_or_not,graph,node_file,input_sample)
    #print(adj)
    #exit()
    splits=StratifiedKFold(n_splits=10,shuffle=True,random_state=1234)

    #print(np.array(features).shape)
    #exit()
    epochs = epochs
    total_num=len(labels)
    o1=open(outfile1,'w+')
    fn=0
    for train_idx,val_idx in splits.split(np.array(features_train),np.array(labels_train)):
        #print('Fold {}'.format(fold+1))
        o1.write('Fold {}'.format(fn+1)+'\n')
        #print(train_idx,val_idx)
        #print(train_idx,val_idx)
        #exit()

        model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,o1)
            test(model,idx_test,features,adj,labels,o1)
        fn+=1
        #test_pred()
    o1.close()
    #o2=open(outfile2,'w+')
    #calculate_avg_acc_of_cross_validation_test.cal_acc_cv(outfile1,outfile2)

####### Species style
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_USA.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_USA_sp_lasso_gcn.txt','Res_record_Sp/r2_USA_sp_lasso_gcn.txt','sample_USA_new.txt')

#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_AUS.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_AUS_sp_lasso_gcn.txt','Res_record_Sp/r2_AUS_sp_lasso_gcn.txt','sample_AUS_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_China.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_China_sp_lasso_gcn.txt','Res_record_Sp/r2_China_sp_lasso_gcn.txt','sample_China_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_Denmark.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_Denmark_sp_lasso_gcn.txt','Res_record_Sp/r2_Denmark_sp_lasso_gcn.txt','sample_Denmark_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_French.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_French_sp_lasso_gcn.txt','Res_record_Sp/r2_French_sp_lasso_gcn.txt','sample_French_new.txt')

### eggNOG style
#run_GCN_test('gcn',500,'Graph_File_test_last_raw/eggNOG_pca_knn_graph_final_trans_Denmark.txt','Node_File/species_node_feature.txt','Res_record_retest_ECE/r1_Denmark_eggNOG_lasso_raw.txt','Res_record_retest_ECE/r2_Denmark_eggNOG_lasso_raw.txt','sample_Denmark_new.txt')



