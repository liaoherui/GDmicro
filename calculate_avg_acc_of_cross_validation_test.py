import re
import os
import numpy as np

def cal_acc_cv(infile,ofile):
    f=open(infile,'r')
    o=open(ofile,'w+')
    dtrain={} # Fold id -> [train_acc]
    dval={} # Fold id -> [val_acc]
    dtest={}
    dtrainu={}
    dvalu={}
    dtestu={}
    

    dtrainm={}
    dvalm={}
    dtestm={}
    dtrainum={}
    dvalum={}
    dtestum={}

    #dtest_acc={}
    #dtest_auc={}
    while True:
        line=f.readline().strip()
        if not line:break
        if not re.search('Fold', line) and not re.search('Epoch',line):continue
        if re.search('Fold',line):
            fid=line.split()[-1]
            dtrain[fid]=[]
            dval[fid]=[]
            dtest[fid]=[]
            dtrainu[fid]=[]
            dvalu[fid]=[]
            dtestu[fid]=[]

            dtrainm[fid]=0
            dvalm[fid]=0
            dtestm[fid]=0
            dtrainum[fid]=0
            dvalum[fid]=0
            dtestum[fid]=0
            #dtest_acc[fid]=0
            #dtest_auc[fid]=0
            continue
        ele=line.split()
        dtrain[fid].append(float(ele[5]))
        dval[fid].append(float(ele[9]))
        dtest[fid].append(float(ele[21]))
        dtrainu[fid].append(float(ele[13]))
        dvalu[fid].append(float(ele[15]))
        dtestu[fid].append(float(ele[23]))

        #ele[-2]=re.sub('accuracy=','',ele[-2])
        #ele[-1]=re.sub('AUC=','',ele[-1])
        '''
        if float(ele[-2])>dtest_acc[fid]:
            dtest_acc[fid]=float(ele[-2])
        if float(ele[-1])>dtest_auc[fid]:
            dtest_auc[fid]=float(ele[-1])
        '''

        if float(ele[5])>dtrainm[fid]:
            dtrainm[fid]=float(ele[5])
        if float(ele[13])>dtrainum[fid]:
            dtrainum[fid]=float(ele[13])
        if float(ele[9])>dvalm[fid]:
            dvalm[fid]=float(ele[9])
        if float(ele[15])>dvalum[fid]:
            dvalum[fid]=float(ele[15])
        if float(ele[21])>dtestm[fid]:
            dtestm[fid]=float(ele[21])
        if float(ele[23])>dtestum[fid]:
            dtestum[fid]=float(ele[23])
    avg_train=[]
    avg_val=[]
    avg_test=[]

    avg_tu=[]
    avg_vu=[]
    avg_teu=[]

    bt_train=[]
    bt_val=[]
    bt_test=[]

    bt_trainu=[]
    bt_valu=[]
    bt_testu=[]

    #bt_tacc=[]
    #bt_tauc=[]
    for i in dtrain:
        #o.write('The average train acc of Fold '+str(i)+' is '+str(np.mean(dtrain[i]))+'\n')
        #o.write('The average train AUC of Fold '+str(i)+' is '+str(np.mean(dtrainu[i]))+'\n')
        o.write('The best train acc of Fold '+str(i)+' is '+str(dtrainm[i])+'\n')
        o.write('The best train AUC of Fold '+str(i)+' is '+str(dtrainum[i])+'\n')
        o.write('The best val acc of Fold '+str(i)+' is '+str(dvalm[i])+'\n')
        o.write('The best val AUC of Fold '+str(i)+' is '+str(dvalum[i])+'\n')
        o.write('The best test acc of Fold '+str(i)+' is '+str(dtestm[i])+'\n')
        o.write('The best test AUC of Fold '+str(i)+' is '+str(dtestum[i])+'\n')
        #o.write('The best test acc of Fold '+str(i)+' is '+str(dtest_acc[i])+'\n')
        #o.write('The best test AUC of Fold '+str(i)+' is '+str(dtest_auc[i])+'\n')
        
        #print('The average train acc of Fold ',i,' is ',np.mean(dtrain[i]))
        #print('The average train AUC of Fold ',i,' is ',np.mean(dtrainu[i]))
        print('The best train acc of Fold '+str(i)+' is '+str(dtrainm[i])+'\n')
        print('The best train AUC of Fold '+str(i)+' is '+str(dtrainum[i])+'\n')

        #print('The average val acc of Fold ',i,' is ',np.mean(dval[i]))
        #print('The average val AUC of Fold ',i,' is ',np.mean(dvalu[i]))
        print('The best val acc of Fold '+str(i)+' is '+str(dvalm[i])+'\n')
        print('The best val AUC of Fold '+str(i)+' is '+str(dvalum[i])+'\n')

        print('The best test acc of Fold '+str(i)+' is '+str(dtestm[i])+'\n')
        print('The best test AUC of Fold '+str(i)+' is '+str(dtestum[i])+'\n')

        #print('The best test acc of Fold '+str(i)+' is '+str(dtest_acc[i])+'\n')
        #print('The best test AUC of Fold '+str(i)+' is '+str(dtest_auc[i])+'\n')





        #o.write('The average val acc of Fold '+str(i)+' is '+str(np.mean(dval[i]))+'\n\n')
        #o.write('The average val AUC of Fold '+str(i)+' is '+str(np.mean(dvalu[i]))+'\n\n')
        #o.write('The best val acc of Fold '+str(i)+' is '+str(dvalm[i])+'\n')
        #o.write('The best val AUC of Fold '+str(i)+' is '+str(dvalum[i])+'\n')


        avg_train.append(np.mean(dtrain[i]))
        avg_val.append(np.mean(dval[i]))
        avg_test.append(np.mean(dtest[i]))

        avg_tu.append(np.mean(dtrainu[i]))
        avg_vu.append(np.mean(dvalu[i]))
        avg_teu.append(np.mean(dtestu[i]))

        bt_train.append(dtrainm[i])
        bt_val.append(dvalm[i])
        bt_test.append(dtestm[i])

        bt_trainu.append(dtrainum[i])
        bt_valu.append(dvalum[i])
        bt_testu.append(dtestum[i])
        #bt_tacc.append(dtest_acc[i])
        #bt_tauc.append(dtest_auc[i])
    '''
    o.write('Final: The average train acc is '+str(np.mean(avg_train))+'\n')
    o.write('Final: The average train AUC is '+str(np.mean(avg_tu))+'\n')
    o.write('Final: The average val acc is '+str(np.mean(avg_val))+'\n')
    o.write('Final: The average val AUC is '+str(np.mean(avg_vu))+'\n')
    '''
    o.write('Final: The averaga train acc is '+str(np.mean(bt_train))+'\n')
    o.write('Final: The average train AUC is '+str(np.mean(bt_trainu))+'\n')
    o.write('Final: The average val acc is '+str(np.mean(bt_val))+'\n')
    o.write('Final: The average val AUC is '+str(np.mean(bt_valu))+'\n')
    o.write('Final: The average test acc is '+str(np.mean(bt_test))+'\n')
    o.write('Final: The average test AUC is '+str(np.mean(bt_testu))+'\n')

    #o.write('Final: The average test acc is '+str(np.mean(bt_tacc))+'\n')
    #o.write('Final: The average test AUC is '+str(np.mean(bt_tauc))+'\n')

    print('The average train acc is ',np.mean(bt_train))
    print('The average train AUC is ',np.mean(bt_trainu))
    print('The average val acc is ',np.mean(bt_val))
    print('The average val AUC is ',np.mean(bt_valu))
    print('The average test acc is ',np.mean(bt_test))
    print('The average test AUC is ',np.mean(bt_testu))
    #print('The average test acc is ',np.mean(bt_tacc))
    #print('The average test AUC is ',np.mean(bt_tauc))

#cal_acc_cv('test.o')
#cal_acc_cv('CRC_41_GCN/FRA/Res_File/r1.txt','CRC_41_GCN/FRA/Res_File/r2.txt')
            
