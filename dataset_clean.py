import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
dataset =pd.read_csv('IoT Network Intrusion Dataset.csv')
data = dataset.copy()
data
data.head()
print("Nbr of raws : " ,len(data))
print("nbr of cols : ", len(data.columns))
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
data.head()
data
data.drop(["Src_IP","Dst_IP","Src_Port","Dst_Port","Timestamp","Protocol","Flow_ID"],axis=1,inplace=True)
data['Label'].value_counts()
data['Cat'].value_counts()
group = {'Anomaly':1,'Normal':0}

data['Label_Num'] = data['Label'].map(lambda x: group[x])
data['Label_Num'].value_counts()
group2 = {'Mirai': 1,
                'Scan': 2,
                'DoS': 3,
                'MITM ARP Spoofing': 4,
               'Normal' : 0}

data['Cat_Num'] = data['Cat'].map(lambda x: group2[x])
data['Cat_Num'].value_counts()
group3 =        { 'Mirai-UDP Flooding' : 1,
                 'Mirai-Hostbruteforceg' : 2,
                 'DoS-Synflooding' : 3,
                 'Mirai-HTTP Flooding' : 4,
                 'Mirai-Ackflooding' : 5,
                 'Scan Port OS' : 6,
                 'MITM ARP Spoofing' : 7,
                 'Scan Hostport' : 8,
                 'Normal' : 0}

data['Sub_Cat_Num'] = data['Sub_Cat'].map(lambda x: group3[x])
data['Sub_Cat_Num'].value_counts()
data.drop(["Label","Cat","Sub_Cat"],axis=1,inplace=True)
data.to_csv("./Datasets/dataset_cleaned.csv", index = False)

