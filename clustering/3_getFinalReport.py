import pickle
import glob

def read_log(path):
    ret={"precision":{}}
    for line in open(path,"r",encoding="utf-8"):
        p1,p2=line.strip().split("total")
        if p1.find("predict")!=-1:
            predict=p1.split(":")[-1].split(",")[0]
            real=p2.split(":")[2].split(",")[0]
            if predict==real:
                ret["precision"][predict]=float(line.strip().split(":")[-1])
        if p1.find("correct")!=-1:
            ret["overall"]=float(line.strip().split(":")[-1])
        if p1.find("reject")!=-1:
            ret["reject"]=float(line.strip().split(":")[-1])
    if "reject" not in ret:
        ret["reject"]=0
    return ret
if __name__=="__main__":
    log_list=glob.glob("acc/*")
    t={}
    for log in log_list:
        result,which=log.split("/")
        this_detail=read_log(log)
        if which not in t:
            t[which]={}
        try:
            t[which]=this_detail
        except:
            print(t)
            print(log)
            exit()
    with open("final_result.pkl","wb")as f:
        pickle.dump(t,f)
