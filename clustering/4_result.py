import pickle
import csv 
if __name__=="__main__":
    with open("final_result.pkl","rb")as f:
        data=pickle.load(f)
    header = ['method','low','medium','high','overall','reject']
    da = []
    for one in data.keys():
        precision = data[one]['precision']
        overall = data[one]['overall']
        reject = data[one]['reject']
        data_lst = ['low','medium','high']
        list_keys = [k for k,v in precision.items()]
        for i in data_lst:
            if i not in list_keys:
                precision[i] = 0
        data_low = precision['low']
        data_med = precision['medium']
        data_high = precision['high']
        add = [data_low,data_med,data_high, overall, reject]
        fin = [one] + add
        da.append(fin)

    file_name = 'acc/'+'result' + '.csv'
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(da)
