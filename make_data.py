import numpy as np 
import matplotlib.pyplot as plt
import sys
import json
import random
import copy
import os
random.seed(123)
def get_mei2id(titles):
    ret={}
    for mei in titles:
        mei_lst=mei.split("+")
        for m in mei_lst:
            if m not in ret:
                ret[m]=len(ret)+1
    return ret

def tolabel(y):
    if y <0.25:
        label="low"
    elif y<0.3:
        label="medium"
    else:
        label="high"
    return label
    
class give_batch():
    def __init__(self, path_raw):
        self.path=path_raw
        self.read_data()
        self.get_label2id()
        self.save_all()
        
    def save_all(self):
        with open("middle_result/mei2id.txt","w",encoding="utf-8")as f:
            for m,id in self.mei2id.items():
                f.write("%s\t%s\n"%(m,id))
        with open("middle_result/label2id2cnt.txt","w",encoding="utf-8")as f:
            for l,id_ in self.label2id2cnt.items():
                id=id_["id"]
                cnt=id_["cnt"]
                f.write("%s\t%s\t%s\n"%(l,id,cnt))
        with open("middle_result/train.txt","w",encoding="utf-8")as f:
            for x,y in zip(self.train_x,self.train_y):
                f.write("%s\t%s\n"%("<=>".join([str(xx) for xx in x]),y))
                
    def read_data(self):
        self.all_data=[]
        for ll,line in enumerate(open(self.path,"r",encoding="utf-8")):
            if ll==0:
                self.titles=line.strip().split("\t")[5:]   
                self.mei2id=get_mei2id(self.titles)        
                continue
            contents=[x.strip() for x in line.split("\t")]
            c1,c2,c3,s1,s2=[float(x) for x in contents[:5]] 
            for jj,mei in enumerate(self.titles):
                y=contents[jj+5]
                y=float(y)
                y=tolabel(y)
                mei_lst=mei.split("+")
                if len(mei_lst)==3:
                    mei_lst=mei_lst+["NO"]
                elif len(mei_lst)==2:
                    mei_lst=mei_lst+["NO","NO"]
                elif len(mei_lst)==1:
                    mei_lst=mei_lst+["NO","NO","NO"]
                self.all_data.append([c1,c2,c3,s1,s2]+mei_lst+[y])
        self.train_x=[xx[:-1]for xx in self.all_data]
        self.train_y=[xx[-1]for xx in self.all_data]
        
    def get_label2id(self):
        self.label2id2cnt={}
        self.id2label={}
        for y in self.train_y:
            if y not in self.label2id2cnt:
                this_id=len(self.label2id2cnt)
                self.id2label[this_id]=y
                self.label2id2cnt[y]={"id":this_id,"cnt":0}
            self.label2id2cnt[y]["cnt"] +=1
    
if __name__ == '__main__':
    if not os.path.exists('middle_result'):
        os.makedirs('middle_result')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    D=give_batch("./data/data_raw")
    print('Load data complete')
