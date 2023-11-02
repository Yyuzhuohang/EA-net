import tensorflow as tf 
import logging
import numpy as np 
import model
import matplotlib.pyplot as plt
import sys
from util import give_batch,get_best_model_ckpt,get_test_acc,\
to_one_hot,plt_hot_confusion,get_result,manully_get_result,write_ceshi_result,to_one_hot_
import config as C
import time    
import os
import csv
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
tf.get_logger().setLevel('ERROR') 
line_num=int(multiprocessing.cpu_count()*0.5)
def show_parameter_count(variables):
    print("---------------Statistical model parameters--------------")
    total_parameters = 0
    for variable in variables:
        name = variable.name
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
            print('{}: {} ({} parameters)'.format(name,shape,variable_parametes))
            total_parameters += variable_parametes
    to_print='Total: {} parameters'.format(total_parameters)
    print(to_print)
    return to_print
class the_net():
    def __init__(self,train_config):
        for item,value in train_config.items():
            print(item)
            print(value)
        self.logger = self.log_config()
        self.save_path=train_config['CKPT']       
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size=train_config["BATCHSIZE"]
        self.max_iter=train_config["MAX_ITER"]
        self.step_save=train_config["STEP_SAVE"]  
        self.step_show=train_config['STEP_SHOW']      
        self.global_steps = tf.Variable(0,trainable=False)  
        self.early_stop=train_config['early_stop'] 
        self.save_best_ckpt=train_config['save_best_ckpt'] 
        if self.save_best_ckpt:
            self.best=-1
            self.best_test=-1
        self.best_acc_tes=0. 
        self.last_improved= 0 
        self.require_improvement = train_config['STOP_THRESHOLD'] 
        self.train_keep_prob=C.train_config["train_keep"]
        self.test_keep_prob=C.train_config["test_keep"]
        self.D=give_batch()
        self.id2label=self.D.id2label
        self.id2mei=self.D.id2mei
        self.sess=tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=line_num,
            intra_op_parallelism_threads=line_num,
        ))
        self.build_net()
        self.build_opt()
        self.saver=tf.train.Saver(max_to_keep=2)
        self.saver_2=tf.train.Saver(max_to_keep=1)
        self.initialize()
        
    def build_net(self):
        self.y = model.neural_network()
        self.loss=self.y.loss
    
    def build_opt(self):
        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(self.loss,global_step=self.global_steps)
    def initialize(self):             
        print("Initialization, I'd like to see more")
        ckpt=tf.train.latest_checkpoint(self.save_path)     
        if ckpt!=None:                                      
            self.saver.restore(self.sess,ckpt)             
        else:
            self.sess.run(tf.global_variables_initializer()) 
    def log_config(self):
        logger = logging.getLogger("log.txt")
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    def log(self,to_log):
        self.logger.info(to_log)
    def get_trainable_variables(self):       
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  
    def train(self):
        st=time.time()
        best_acc_tes=0. 
        last_improved= 0 
        for step in range(self.max_iter):
            x,y=self.D.do(self.batch_size)
            y=[to_one_hot(yyy)for yyy in y]
            c=[xx[:3]for xx in x]
            s=[xx[3:5]for xx in x]
            mei=[xx[5:]for xx in x]
            mei_onehot=self.to_one_hot(mei)
            loss,_,gs,out=self.sess.run([self.loss,self.opt,self.global_steps,self.y.out_1],\
                    feed_dict={self.y.c_input:c,self.y.s_input:s,self.y.mei_input:mei,self.y.label:y,\
                               self.y.mei_onehot:mei_onehot,\
                               self.y.keep_prob:self.train_keep_prob}) 
            if (step+1)%self.step_show==0:
                to_print="loss %s,in global step %s, \
                             taks %s seconds"%\
                            (loss,gs,time.time()-st)
                mei_onehot=self.to_one_hot(mei)
                self.logger.info(to_print)
                print(to_print)
                st=time.time()                 
                if self.save_best_ckpt or self.early_stop:
                    if self.save_best_ckpt:
                        if acc_now > self.best:
                            self.saver.save(self.sess, self.save_path+"_train"+"/best_check_%s.ckpt"%(str(gs)))
                            self.best=acc_now
                            print("update the ckpt_train at accuracy %s"%self.best)
                            self.log("update the ckpt_train at accuracy %s"%self.best)
                            continue
                        elif acc > self.best_test:
                            self.saver_2.save(self.sess, self.save_path+"_test"+"/best_check_%s.ckpt"%(str(gs)))
                            self.best_test=acc
                            print("update the ckpt_test at accuracy %s"%self.best_test)
                            self.log("update the ckpt_test at accuracy %s"%self.best_test)
                            continue
                        continue
                    else:
                        f=0
                        if acc > best_acc_tes+0.005:
                            best_acc_tes=acc
                            if f!=0:
                                os.remove(self.save_path+"/best_check_%s.ckpt"%(str(last_improved)))
                            last_improved= gs
                            self.saver.save(self.sess, self.save_path+"/best_check_%s.ckpt"%(str(gs)))
                            f+=1
    
                        if gs-last_improved > self.require_improvement:
                            to_print="(Early stopping in %s step! And the best ceshi accuracy is %s.)"%(last_improved,best_acc_tes)
                            print(to_print)
                            self.logger.info(to_print)
                            break
                else: 
                    if (step+1)%self.step_save==0:
                        self.saver.save(self.sess, self.save_path+"/check.ckpt")
         
    def get_acc(self,y,predict,name="Test==>"):
        total=len(y)
        right=0
        for real,pre in zip(y,predict):
            real=list(real).index(max(real))
            pre=list(pre).index(max(pre))
            if real==pre:
                right+=1
        acc = right/total
        to_print=name+" total: %s, right: %s, ratio: %s"%(total,right,acc)
        print(to_print)
        self.logger.info(to_print)
        return acc
    def to_one_hot(self,mei):
        ret=[]
        for mm in mei:
            tmp=[0 for i in range(self.y.mei_num)]
            for m in mm:
                tmp[m]=1.0
            ret.append(tmp)
        return ret
    def ceshi(self,Flag=False):
        x=self.D.ceshi_x_mask
        y=self.D.ceshi_y_mask
        y=[to_one_hot(yyy)for yyy in y]
        c=[xx[:3]for xx in x]
        s=[xx[3:5]for xx in x]
        mei=[xx[5:]for xx in x]
        mei_onehot=self.to_one_hot(mei)
        predict=self.sess.run(self.y.predict,\
        feed_dict={self.y.c_input:c,self.y.s_input:s,self.y.mei_input:mei,\
                    self.y.mei_onehot:mei_onehot,\
                    self.y.label:y,self.y.keep_prob:1.0})        
        ceshi_acc = self.get_acc(y,predict,name="CESHI==>")
        write_ceshi_result(c,s,mei,y,predict,self.id2mei,self.id2label)  
        return ceshi_acc
def test(ckpt_path):
    D=give_batch()
    x=D.ceshi_x_mask
    y_=D.ceshi_y_mask
    y=[to_one_hot(yyy)for yyy in y_]
    c=[xx[:3]for xx in x]
    s=[xx[3:5]for xx in x]
    mei=[xx[5:]for xx in x]
    id2label_dict=D.id2label
    id2mei_dict=D.id2mei
    net = model.neural_network()
    best_model = get_best_model_ckpt(ckpt_path)
    print("\nThe best model is:%s"%best_model)
    print("+++++++++++++++++++++Start loading the model++++++++++++++++++++\n")
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, best_model)
        mei_onehot=to_one_hot_(mei)
        encode=sess.run(net.out_,\
                              feed_dict={net.c_input:c,net.s_input:s,net.mei_input:mei,\
                                         net.mei_onehot:mei_onehot,\
                                         net.label:y,net.keep_prob:1.0}
                              )
        with open('encode.txt','w',encoding="utf-8")as f:
            for xx,yy,line in zip(x,y_,encode):
                cs = xx[:-4]
                cs = [str(cs_) for cs_ in cs]
                mei_ = xx[5:]
                mei_1 = [id2mei_dict[x_] for x_ in mei_]
                yy = id2label_dict[yy]
                sample = '<=>'.join(cs+mei_1)
                lines = ' '.join([str(l) for l in line])
                result = '\t'.join([sample,yy,lines])
                f.write(result+'\n')
        f.close()

if __name__=="__main__":
    if sys.argv[1] == 'train':
        main_net=the_net(C.train_config)
        all_variable=main_net.get_trainable_variables()
        to_print=show_parameter_count(all_variable)
        main_net.log(to_print)
        main_net.train()
    if sys.argv[1] == 'test':
        test("ckpt")    