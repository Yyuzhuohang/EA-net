import numpy as np 
import matplotlib.pyplot as plt
import sys
import json
import random
import glob
import config as C
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report 
import pandas as pd
class give_batch():
    def __init__(self):
        self.check_and_load()
    def split_data(self,info,make_label=False):
        if make_label:
            xx=info.strip('\n').split("<=>")
            mei=xx[-4:]
            cs=[float(xxx)for xxx in xx[:-4]]
            return cs,mei
        else:
            x,y=info.strip().split("\t")
            xx=x.split("<=>")
            mei=xx[-4:]
            cs=[float(xxx)for xxx in xx[:-4]]           
            return cs,mei,y
    def check_and_load(self):
        if glob.glob("middle_result/mei2id.txt")==[]\
        or glob.glob("middle_result/label2id2cnt.txt")==[]\
        or glob.glob("middle_result/train.txt")==[]:
            print("raw data wrong! please run 0_make_data.py")
            exit()

        self.mei2id={}
        self.id2mei={0:'None'}
        for line in open("middle_result/mei2id.txt","r",encoding="utf-8"):
            m,id=line.strip().split("\t")
            self.id2mei[int(id)]=m
            self.mei2id[m]=int(id)
        self.label2id={}
        self.id2label={}
        for line in open("middle_result/label2id2cnt.txt","r",encoding="utf-8"):
            l,id,cnt=line.strip().split("\t")
            self.label2id[l]=int(id)
            self.id2label[int(id)]=l
        self.train_x=[]
        self.train_y=[]
        self.train_dict={}
        for line in open("middle_result/train.txt","r",encoding="utf-8"):
            cs,mei,y = self.split_data(line)
            self.train_x.append(cs+mei) 
            self.train_y.append(y) 
            if y not in self.train_dict:
                self.train_dict[y]={"sample":[],"num":[]}
            self.train_dict[y]["sample"].append(cs+mei)
            self.train_dict[y]["num"].append(len(self.train_dict[y]["num"]))   
        self.index_pool=[i for i in range(len(self.train_x))] 
            
        if glob.glob("middle_result/train.txt")!=[]:
            self.ceshi_x=[]
            self.ceshi_y=[]
            for line in open("middle_result/train.txt","r",encoding="utf-8"):
                cs,mei,y = self.split_data(line)
                self.ceshi_x.append(cs+mei)
                self.ceshi_y.append(y)
                
        self.ceshi_x_mask,self.ceshi_y_mask=self.mask(self.ceshi_x,self.ceshi_y)
        self.train_x_mask,self.train_y_mask=self.mask(self.train_x,self.train_y)
        self.num_class=len(self.train_dict)
        
    def do(self,batchsize):
        each=int(batchsize/self.num_class)
        x=[]
        y=[]
        for yy,xx_s in self.train_dict.items():
            this_x_lst=xx_s["sample"]
            index_pool=xx_s["num"]
            this_index=np.random.choice(index_pool,[each])
            
            for ind in this_index:
                x.append(this_x_lst[ind])
                y.append(yy)
        x,y=self.mask(x,y)
        x = self.creat_data(x)
        return x,y
    def do_(self,batchsize):
        this_index=np.random.choice(self.index_pool,[batchsize])
        x=[]
        y=[]
        for index in this_index:
            x.append(self.train_x_mask[index])
            y.append(self.train_y_mask[index])
        return x,y
    def mask(self,x,y,make_label=False):
        ret_x=[]
        ret_y=[]
        for xx in x:
            cs=xx[:-4]
            mei_lst=[self.mei2id.get(i,0)for i in xx[-4:]]
            ret_x.append(cs+mei_lst)
        for yy in y:
            ret_y.append(self.label2id[yy])
        return ret_x,ret_y
    def creat_data(self,x):
        ret_xx = []
        for index_x in x:
            s_c = index_x[:5]
            mei = index_x[5:]
            c_s = []
            for c in s_c:
                cc = c+c*0.01
                c_s.append(round(cc,3))
            xx = c_s+mei
            ret_xx.append(xx)
        return x+ret_xx
        
def get_best_model_ckpt(path):
    ckpt_list_tf = tf.train.get_checkpoint_state(path+'/')
    for ckpt in ckpt_list_tf.all_model_checkpoint_paths:
        if 'best' in ckpt:
            best_model = ckpt
            break
        else:
            best_model = ckpt
    return best_model
    
def get_test_acc(y,predict):
    total=len(y)
    right=0
    for real,pre in zip(y,predict):
        real=list(real).index(max(real))
        pre=list(pre).index(max(pre))
        if real==pre:
            right+=1
    acc = right/total
    return acc
    
def to_one_hot(y_num):
    ret=[0]*C.stru_config["num_class"]
    ret[y_num]=1
    return ret
    
def to_one_hot_(mei):
        ret=[]
        for mm in mei:
            tmp=[0 for i in range(13)]
            for m in mm:
                tmp[m]=1.0
            ret.append(tmp)
        return ret
        
def get_confusion_matrix(con,label):
    confusion_matrix=np.zeros([len(label),len(label)])
    for tru,pre_dict in con.items():
        if tru=='low':
            i=0
            j=0
            for label_index in label:
                confusion_matrix[i][j]=pre_dict[label_index]
                j+=1
        if tru=='medium':
            i=1
            j=0
            for label_index in label:
                confusion_matrix[i][j]=pre_dict[label_index]
                j+=1
        if tru=='high':
            i=2
            j=0
            for label_index in label:
                confusion_matrix[i][j]=pre_dict[label_index]
                j+=1
    return confusion_matrix
    
def plt_hot_confusion(con,label,flag='sk'):
    if flag=='manully':
        confusion_matrix = get_confusion_matrix(con,label)
        plt.imshow(confusion_matrix, cmap=plt.cm.coolwarm)
        indices = range(len(con.keys()))
        plt.xticks(indices, label,fontsize=12)
        plt.yticks(indices, label,fontsize=12)
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=16)
        plt.title('The confusion matrix of the EA-net method',fontsize=15)
        plt.xlabel('Prediction',fontsize=12)
        plt.ylabel('True',fontsize=12)
        for first_index in range(len(confusion_matrix)):
            for second_index in range(len(confusion_matrix[first_index])):
                plt.text(second_index, first_index,confusion_matrix[first_index][second_index],fontsize=16)
        plt.savefig('confusion.jpg', dpi=500)
    else:
        plt.imshow(con, cmap=plt.cm.Blues)
        indices = range(len(con))
        plt.xticks(indices, label)
        plt.yticks(indices, label)
        plt.colorbar()
        plt.xlabel('Prediction')
        plt.ylabel('True')
        for first_index in range(len(con)):
            for second_index in range(len(con[first_index])):
                plt.text(first_index, second_index, con[first_index][second_index])
        plt.savefig('confusion.jpg', dpi=300)
        
def get_result(true,pred,id2label_dict):
    true_list = [x.index(1) for x in true]
    pred_list = [list(y).index(max(list(y))) for y in pred]
    target_names=['low','medium','high']
    classification_info = classification_report(true_list,pred_list,target_names=target_names,output_dict=True)
    print(classification_info)
    df = pd.DataFrame(classification_info)#.transpose()
    df.to_csv("classification_result.csv",index=True)
    print("The results of the model evaluation have been saved classification_result.csv")
    confusion = confusion_matrix(true_list,pred_list)
    print("The confusion matrix is shown below \n %s"%confusion)
    plt_hot_confusion(confusion,target_names,'sk')
    
def manully_get_result(true,pred,id2label_dict):
    confusion_dict={}
    target_names=['low','medium','high']
    for x,y in zip(true,pred):
        true_label,pred_label=id2label_dict[x.index(1)],id2label_dict[list(y).index(max(list(y)))]
        if true_label not in confusion_dict and true_label=='low':
            confusion_dict[true_label]={}
        if true_label not in confusion_dict and true_label=='medium':
            confusion_dict[true_label]={}
        if true_label not in confusion_dict and true_label=='high':
            confusion_dict[true_label]={}
        if pred_label not in confusion_dict[true_label] and pred_label=='low':
            confusion_dict[true_label][pred_label]=0
        if pred_label not in confusion_dict[true_label] and pred_label=='medium':
            confusion_dict[true_label][pred_label]=0
        if pred_label not in confusion_dict[true_label] and pred_label=='high':
            confusion_dict[true_label][pred_label]=0
        confusion_dict[true_label][pred_label]+=1
    for label,pre_l in confusion_dict.items():
        flag = [[x in pre_l.keys(),idex] for idex,x in enumerate(target_names)]
        for x in flag:
            if False==x[0]:
                i,j=label,x[-1]
                confusion_dict[i][target_names[j]]=0
    plt_hot_confusion(confusion_dict,target_names,'manully')
    
def write_ceshi_result(c,s,mei,y,predict,id2mei_dict,id2label_dict):
    with open('ceshi_result',"w",encoding="utf-8")as f:
        for c1,s1,mei1,yy,pp in zip(c,s,mei,y,predict):
            mei_name = [id2mei_dict[x] for x in mei1]
            y_name = id2label_dict[yy.index(1)]
            p_name = id2label_dict[list(pp).index(max(list(pp)))]
            content_input="<=>".join([str(x) for x in c1]+[str(x) for x in s1]+mei_name)
            content_output="\t".join([content_input,y_name,p_name+'\n'])
            f.write(content_output)
    f.close()
