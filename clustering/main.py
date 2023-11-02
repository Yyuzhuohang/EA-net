import config as CF
import model as m
import glob
import os
import time
import numpy as np

def load_data(file_):
    data=[]
    file_lst=[]
    for line in open(file_,"r",encoding="utf-8"):
        name,label,da =line.strip().split("\t")
        label_='\t'.join([name,label])
        file_lst.append(label_)
        da_=[]
        for x in da.split(" "):
            try:
                da_.append(float(x))
            except:
                da_.append(0.0)
        data.append(da_)
    return file_lst,data
st=time.time()
file_lst,data=load_data("raw_data")
print("sample number is %s, sample dim is %s"%(len(data),len(data[0])))
class_num=CF.config["class_num"]
for model_type in CF.config["type"]: 
    model= m.model(class_num=class_num)
    model.build(model_type)
    model.run(data)
    model.show(file_lst)

print("Clustering is complete, when it is shared:%s秒"%(time.time()-st))
