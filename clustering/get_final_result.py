import config as CF
import glob
import os,random
def get_first_label(file_path):
    files=glob.glob(file_path+"/*")
    ret_im2label={}
    ret_label2im={}
    for files_ in files:
        label=files_.split("/")[-1].split('.')[0]
        if label not in ret_label2im:
            ret_label2im[label]=[]
        for ims in open(files_,'r',encoding="utf-8"):
            ims=ims.strip()
            ret_im2label[ims]=label
            ret_label2im[label].append(ims)
    return ret_im2label,ret_label2im
    
def get_max(t):
    max_=""
    max_v=0
    for key,value in t.items():
        if value >max_v:
            max_=key
            max_v=value
    return max_

def get_non_first_label(t_first_im2label,file_path):
    files=glob.glob(file_path+"/*")
    ret_label2im={}
    for files_ in files:
        t_count={}
        tmp=[]
        for ims in open(files_,'r',encoding="utf-8"):
            ims=ims.strip()
            label=t_first_im2label[ims]
            t_count[label]=t_count.get(label,0)+1
            tmp.append(ims)
        this_label=get_max(t_count)
        ret_label2im[this_label]=tmp
        tmp=[]
    return ret_label2im

def keep(file_path):
    files=glob.glob(file_path+"/*")
    if len(files)>0 and len(files)<=class_num:
        return True
    else:
        return False
def make_blank_dict(raw_path):
    t={}
    raw_file_lst=glob.glob(raw_path+"/*")
    for raw in raw_file_lst:
        t[raw]={}
    print("there are %s clusters"%len(t))
    return t
def inter(lst1,lst2):
    ret=[]
    for x in lst1:
        if x in lst2:
            ret.append(x)
    return ret

if __name__=="__main__":
    class_num=CF.config["class_num"]
    t_whole={}
    files=glob.glob("result/*")
    for fi in files:
        if len(glob.glob("%s/*"%fi))==class_num:  
            print("the first chosen file is %s"%fi) 
            t_im2label_first,t_label2im_first=get_first_label(fi)
            files.remove(fi)
            t_whole[fi]=t_label2im_first 
            break
        elif not keep(fi):
            files.remove(fi)
    
    for fi in files:
        if keep(fi):
            t_label2im_=get_non_first_label(t_im2label_first,fi)
            t_whole[fi]=t_label2im_
        else:
            print("abandom %si"%(fi))

    t_final=t_label2im_first
    print("There are a total of %s clustering methods"%(len(t_whole)))
    for fi,items in t_whole.items():
        for label,ims in items.items():
            if label not in t_final:
                t_final[label]=[]
            print("Now deal with the clustering method %s. Before the union, the category %s has %s samples"%(fi,label,len(t_final[label])))
            t_final[label]=inter(t_final[label],ims)
            print("After taking the union, the category %s has %s samples"%(label,len(t_final[label])))
    if glob.glob("result_final")==[]:
        os.system("mkdir result_final")
    
    all_img_path=CF.all_path
    ims_all = []
    dict_mei = {'NO':'None'}
    for x in open(all_img_path,'r',encoding="utf-8"):
        x,y = x.strip().split('\t')
        xx = x.split('<=>')
        cs = xx[:5]
        mei = xx[5:]
        mei_ = []
        for x_ in mei:
            if x_=='NO':
                x_ = dict_mei[x_]
            mei_.append(x_)
        xxx = cs + mei_
        xxx = '<=>'.join(xxx)
        final_x = '\t'.join([xxx,y])
        ims_all.append(final_x)
    total_num_pic=len(ims_all)
    
    for label,ims in t_final.items():
        path_t="result_final/bagging/%s.txt"%label
        if glob.glob(path_t)!=[]:
            os.system("rm %s && %s"%(path_t,path_t))
        with open(path_t,'w',encoding="utf-8")as f:
            for im in ims:
                ims_all.remove(im)
                f.write(im+'\n')
    print("total number is %s, the number of well classified is %s"%(total_num_pic,len(ims_all)))
    rest_file_path="result_final/bagging/rest.txt"
    if glob.glob(rest_file_path)!=[]:
        os.system("rm %s"%rest_file_path)
    with open(rest_file_path,'w',encoding="utf-8")as f1:
        for im in ims_all:
            f1.write(im+'\n')
