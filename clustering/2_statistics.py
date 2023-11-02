import glob
import config as C

def get_dominate(folder):
    labels=[x.strip().split("\t")[-1] for x in open(folder,'r',encoding="utf-8")]
    t_label2cnt={}
    for label in labels:
        t_label2cnt[label]=t_label2cnt.get(label,0)+1
    return t_label2cnt,len(labels)

def do_one_model(path):
    label_files=glob.glob(path+"/*")
    t_pre2cor={}
    t_pre2cnt={}
    for pre_label_file in label_files:
        if pre_label_file.find("rest.txt")!=-1:continue
        t_label2cnt,this_total=get_dominate(pre_label_file)
        try:
            this_label=[x[0] for x in sorted(t_label2cnt.items(),key=lambda x:x[1],reverse=True)][0]
        except:
            print(t_label2cnt)
            return "no","no"
        t_pre2cnt[this_label]=t_pre2cnt.get(this_label,0)+this_total
        for real_label,cnt in t_label2cnt.items():
            title="%s->%s"%(this_label,real_label)
            t_pre2cor[title]=t_pre2cor.get(title,0)+cnt
    return t_pre2cor,t_pre2cnt

if __name__=="__main__":
    root_folder_lst=glob.glob("result/*")
    for root_folder in root_folder_lst:
        path_lst=glob.glob('result'+"/*")
        for path in path_lst:
            if path.find("bagging")!=-1:
                num_re = len([x.strip() for x in open(path+"/rest.txt",'r',encoding="utf-8")])
            else:
                num_re=0
            the_name = path.split('/')[-1]

            with open("acc/"+the_name,"w",encoding="utf-8")as f:
                t_pre2cor,t_pre2cnt=do_one_model(path)
                if t_pre2cor=="no":
                    continue
                pre_total={}
                correct_num=0
                for p_c,cnt in t_pre2cor.items():
                    p,c=p_c.split("->")
                    this_pre_total=t_pre2cnt[p]
                    if p not in pre_total:
                        pre_total[p] =this_pre_total
                    if p==c:
                        correct_num +=cnt
                    f.write("predict:%s,total:%s,real:%s,number:%s,\
                        ratio:%.4f\n"%(p,this_pre_total,c,cnt,cnt/this_pre_total))
                pre_total=sum([x[1]for x in pre_total.items()])
                
                f.write("correct:%s,total number of sample:%s,overall accuracy:%.4f\n"%(correct_num,pre_total,correct_num/(pre_total)))
                if path.find("bagging")!=-1:
                    f.write("reject:%s, total number of sample:%s, reject rate:%.4f\n"%(num_re,pre_total +num_re,num_re/(pre_total+num_re)))
