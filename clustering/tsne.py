import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
def load_data(file_path):
    label_lst=[]
    data_lst=[]
    for info in open(file_path,'r',encoding="utf-8"):
        info = info.strip().split('\t')
        label_lst.append(info[1])
        encode = info[2].split(' ')
        encode = [str(x) for x in encode]
        data_lst.append(encode)
        data_np = np.array(data_lst)
    return label_lst, data_np

def plot_embedding(data, label, flag):
    color_dict={'low':'red','medium':'yellow','high':'blue'}
    maker_dict={'low':'+','medium':'o','high':'v'}
    data_dict={}
    fig = plt.figure()
    for i in range(data.shape[0]):
        if label[i] not in data_dict:
            data_dict[label[i]]=[]
        data_dict[label[i]].append((data[i,0],data[i,1]))

    for label,value in data_dict.items():
        for value_ in value:
            plt.scatter(value_[0],value_[1],color=color_dict[str(label)],marker=maker_dict[str(label)],s=30)
    plt.savefig('t-SNE_%s.png'%(flag))
    
def main():
    file_path = './raw_data'
    label,data = load_data(file_path)

    tsne = TSNE(n_components=2, init='pca',random_state=0)
    result = tsne.fit_transform(data)
    max_,min_ = np.max(result,0),np.min(result,0)
    data = (result - min_)/(max_ - min_)
    
    plot_embedding(data, label, '1')

main()
