import glob
import os


data_path = "result/"
save_path = "right_result/"
label_dic={}
mode_lst = glob.glob(data_path+"*")
for mode_name in mode_lst:
    label_lst = glob.glob(mode_name+"/*")
    m = mode_name.split('/')[-1]
    if m not in label_dic:
        label_dic[m]={}
    for result_name in label_lst:
        for label_name in open(result_name,'r',encoding="utf-8"):
            label = label_name.strip().split("\t")[-1]
            pre = result_name.split('/')[-1].split('.')[0]
            if pre not in label_dic[m]:
                label_dic[m][pre]={}
            label_dic[m][pre][label] = label_dic[m][pre].get(label,0)+1
print(label_dic)
print("++++++++++++++++++++++++++++++++++++++++++++")


