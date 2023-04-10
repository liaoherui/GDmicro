import re
import os

f=open('Rerun_CRC/FRA_dsd/Res_File/sample_kneighbors_all.txt','r')
line=f.readline()
c=0
while True:
    line=f.readline().strip()
    if not line:break
    ele=line.split('\t')
    #print(ele)
    if c>461:break 
    p=0
    n=0
    for e in ele[1:]:
        st=re.split(':',e)[1]
        #print(st)
        #exit()
        if st=='Health':
            n+=1
        else:
            p+=1
    if n>1 and p>1:
        print(line) 
    c+=1

