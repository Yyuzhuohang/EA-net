import tensorflow as tf 
import time
import numpy as np
import config as C
def activation(x,which="tanh"):
    if which =="tanh":
        return tf.nn.tanh(x)
    if which=="relu":
        return tf.nn.relu(x)
    if which=="sigmoid":
        return tf.nn.sigmoid(x)
    if which=="sotfmax":
        return tf.nn.sotfmax(x)
class Contrastive_Loss:
    def __init__(self):
        self.temperature = 20
    def forward(self, features):    
        b, n, dim = features.shape 
        unbind_features = tf.unstack(features,axis=1)
        out_1,out_2 = unbind_features[0],unbind_features[1]
        contrast_features = tf.concat(unbind_features,0)
        sim_matrix = tf.exp(tf.matmul(contrast_features, tf.transpose(contrast_features)) / self.temperature)
        mask = tf.cast(tf.ones_like(sim_matrix)-tf.eye(int(b)*2),tf.bool)
        sim_matrix = tf.reshape(tf.boolean_mask(sim_matrix,mask),(int(b)*2,-1))
        pos_sim = tf.exp(tf.reduce_sum(out_1*out_2,-1) / self.temperature)
        pos_sim = tf.concat([pos_sim,pos_sim],0)
        loss = tf.reduce_mean( - tf.log(pos_sim / tf.reduce_sum(sim_matrix,-1)))
        return loss
class neural_network:
    def __init__(self):
    
        self.mei_num=13 
        self.c_num=3 
        self.s_num=2 
        self.mei_num_fanying=4
        self.num_class=C.stru_config["num_class"] 
        
        self.embedding_size=C.stru_config["embedding_size"]
        self.which=C.stru_config["which"] 
        self.attention_size=C.stru_config["attention_size"] 
        self.batch_ = C.train_config["BATCHSIZE"]
        
        self._bias_initializer=tf.zeros_initializer() 
        self.weight_initialization =  tf.contrib.layers.xavier_initializer()
        self.construct_input()
        self.build_model()
        self.build_res()
        self.build_loss()
    def construct_input(self):
        self.c_input=tf.placeholder(tf.float32,[None, self.c_num])
        self.s_input=tf.placeholder(tf.float32,[None, self.s_num])
        self.mei_input=tf.placeholder(tf.int32,[None, self.mei_num_fanying])
        self.mei_onehot=tf.placeholder(tf.float32,[None, self.mei_num])
        self.keep_prob=tf.placeholder(tf.float32)
        with tf.variable_scope("embedding"):            
            self.W_embedding=tf.get_variable(shape=[self.mei_num,self.embedding_size],\
                    initializer=self.weight_initialization,name="W_embedding")
            self.mei_input_embedding=tf.nn.embedding_lookup(self.W_embedding,self.mei_input)

        self.label=tf.placeholder(tf.int32,[None, self.num_class])
    def build_model(self):    
        with tf.variable_scope("c_forward_1"):
            W = tf.get_variable(shape=[self.c_num,self.embedding_size],
                                       initializer=self.weight_initialization, name="W_filter")
            b = tf.get_variable(shape=[self.embedding_size], initializer=self._bias_initializer, name="bias")
            self.c_tmp=activation(tf.matmul(self.c_input,W)+b,self.which)
            self.c_tmp = tf.nn.dropout(self.c_tmp,self.keep_prob)
        with tf.variable_scope("s_forward_1"):
            W = tf.get_variable(shape=[self.s_num,self.embedding_size],
                                       initializer=self.weight_initialization, name="W_filter")
            b = tf.get_variable(shape=[self.embedding_size], initializer=self._bias_initializer, name="bias")
            self.s_tmp=activation(tf.matmul(self.s_input,W)+b,self.which)
            self.s_tmp = tf.nn.dropout(self.s_tmp,self.keep_prob)
        with tf.variable_scope("attention_K_Q"):
            Qc = tf.get_variable(shape=[self.embedding_size,self.attention_size], 
                                       initializer=self.weight_initialization, name="Qc")
            Kc = tf.get_variable(shape=[self.embedding_size,self.attention_size],
                                       initializer=self.weight_initialization, name="Kc")
            bqc = tf.get_variable(shape=[self.attention_size], initializer=self._bias_initializer, name="bqc")
            bkc = tf.get_variable(shape=[self.attention_size], initializer=self._bias_initializer, name="bkc")
            Qs = tf.get_variable(shape=[self.embedding_size,self.attention_size],
                                       initializer=self.weight_initialization, name="Qs")
            Ks = tf.get_variable(shape=[self.embedding_size,self.attention_size],
                                       initializer=self.weight_initialization, name="Ks")
            bks = tf.get_variable(shape=[self.attention_size], initializer=self._bias_initializer, name="bks")
            bqs = tf.get_variable(shape=[self.attention_size], initializer=self._bias_initializer, name="bqs")
            Q_s=activation(tf.matmul(self.s_tmp,Qs)+bqs,self.which)
            Q_c=activation(tf.matmul(self.c_tmp,Qc)+bqc,self.which)
            K_s=activation(tf.matmul(self.s_tmp,Ks)+bks,self.which)
            K_c=activation(tf.matmul(self.c_tmp,Kc)+bkc,self.which)
            alpha1=tf.reduce_sum(Q_s*Q_c,axis=-1)
            alpha1_lst=[]
            for _ in range(self.embedding_size):
                alpha1_lst.append(alpha1)
            alpha1=tf.stack(alpha1_lst,axis=-1)
            alpha2=tf.reduce_sum(K_s*K_c,axis=-1)
            alpha2_lst=[]
            for _ in range(self.embedding_size):
                alpha2_lst.append(alpha2)
            alpha2=tf.stack(alpha2_lst,axis=-1)
            self.c_s_out_put=alpha1*self.c_tmp+alpha2*self.s_tmp
        self.mei_input_embedding_lst=tf.split(self.mei_input_embedding,num_or_size_splits=self.mei_num_fanying,axis=1)
        coe=[]
        for x in self.mei_input_embedding_lst:   
            x=tf.squeeze(x,axis=1)
            coe_tmp=tf.reduce_sum(x*self.c_s_out_put,axis=-1) 
            coe_tmp_lst=[]
            for _ in range(self.embedding_size):
                coe_tmp_lst.append(coe_tmp)
            coe_tmp=tf.stack(coe_tmp_lst,axis=-1)
            coe.append(coe_tmp)
        self.out=0
        for c,x in zip(coe,self.mei_input_embedding_lst):
            x=tf.squeeze(x,axis=1)
            self.out += c*x
        with tf.variable_scope("final"):
            W = tf.get_variable(shape=[self.embedding_size,self.num_class],
                                       initializer=self.weight_initialization, name="f_f")
            b = tf.get_variable(shape=[self.num_class], initializer=self._bias_initializer, name="bias")
            self.predict_main=tf.matmul(self.out,W)+b

    def build_res(self):
        the_activate="relu"
        self.res_size=C.stru_config["res_size"]
        
        with tf.variable_scope("c_s_m_res"):
            self.this_input=tf.concat([self.c_input,self.s_input,self.mei_onehot],axis=-1)
            
            W_c = tf.get_variable(shape=[self.c_num+self.s_num+self.mei_num,self.res_size],
                                       initializer=self.weight_initialization, name="C_W_filter")
            b = tf.get_variable(shape=[self.res_size], initializer=self._bias_initializer, name="bias")
            self.res_tmp=activation(tf.matmul(self.this_input,W_c)+b,the_activate)
        with tf.variable_scope("res_final"):
            W = tf.get_variable(shape=[self.res_size,self.num_class],
                                       initializer=self.weight_initialization, name="W_filter")
            b = tf.get_variable(shape=[self.num_class], initializer=self._bias_initializer, name="bias")
            self.predict_res = tf.matmul(self.res_tmp,W)+b
        
    def build_loss(self):
        self.out_ = tf.concat([self.out,self.res_tmp],-1)
        b, d = self.out_.shape
        self.out_1 = tf.reshape(self.out_,[self.batch_-1,2,d])
        self.loss_ = Contrastive_Loss()
        self.loss = self.loss_.forward(self.out_1)
