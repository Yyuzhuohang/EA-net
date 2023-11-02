#!-*-coding:utf-8-*-
from sklearn.cluster import KMeans,AgglomerativeClustering,AffinityPropagation,MeanShift,estimate_bandwidth,SpectralClustering,DBSCAN,Birch
from sklearn.datasets._samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import glob
import shutil
import os
import time
class model(object):
    def __init__(self,class_num):
        self.class_num=class_num
    def build(self,model_type):
        self.model_type=model_type
        if self.model_type=="kmeans":
            self.estimator = KMeans(n_clusters=self.class_num)
            print("Choose an algorithm k_means")
        elif self.model_type=="agg":
            self.estimator = AgglomerativeClustering(n_clusters=self.class_num, linkage='ward')
            print("The algorithm Agg is used")
        elif self.model_type=="ap":
            print("Select the algorithmic AP")
            self.estimator= AffinityPropagation()
        elif self.model_type=="mean-shift":
            print("The algorithm mean-shift is used")
            self.estimator=MeanShift()
        elif self.model_type=="spectral":
            print("The algorithm spectral is used")
            self.estimator=SpectralClustering(n_clusters=self.class_num)
        elif self.model_type=="dbscan":
            print("The algorithm dbscan is used")
            self.estimator=DBSCAN()
        elif self.model_type=="gmm":
            print("The algorithm gmm is selected")
            self.estimator=GaussianMixture(n_components=self.class_num)
        elif self.model_type=="birch":
            print("The algorithm BIRCH is selected")
            self.estimator=Birch(threshold=0.11,branching_factor=25,n_clusters =self.class_num)
        else:
            print("wrong model type, please check config")
            exit()
    def run(self,data):
        print("Clustering")
        st=time.time()
        self.estimator.fit(data)
        print("Clustering completes, time %s s"%(time.time()-st))
    def show(self,file_list):
        self.label_pred = self.estimator.labels_
        if glob.glob("result/%s"%self.model_type)==[]:
            os.system("mkdir result/%s"%self.model_type)
        for label in set(self.label_pred):
            if glob.glob("result/%s"%(self.model_type))!=[]:
                os.system("rm -r result/%s/%s.txt"%(self.model_type,label))
        for file_path,label in zip(file_list,self.label_pred):
            with open("result/%s/%s.txt"%(self.model_type,label),'a',encoding="utf-8")as f:
                f.write(file_path+'\n')
