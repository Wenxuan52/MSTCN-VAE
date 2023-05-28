import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Path = './/data//iMiGUE//imigue_skeleton_test//'

SampleID = '0001'
maxframecount = []
framecount = []
sample_count = 0
d0 = []
d10 = []
d20 = []
d30 = []
d40 = []
d50 = []
d60 = []
d70 = []
d80 = []
d90 = []
d100 = []
d200 = []
d300 = []
d400 = []
d500 = []
d600 = []
for SampleID in os.listdir(Path):
    if SampleID == '0218' or SampleID == '0319' or SampleID == '0347' or SampleID == '0348':
        continue

    SamplePath = Path + SampleID + '//' + SampleID + '_light_hand.csv'
    SamplelabelPath = Path + SampleID + '//' + SampleID + '_label.csv'
    # print(SamplePath)
    # df = pd.read_csv(SamplePath, header=None)
    df1 = pd.read_csv(SamplelabelPath, header=None)
    # print(df1.shape)
    dfnp = df1.values
    framedistance = dfnp[:, 2] - dfnp[:, 1]
    # print(framedistance)
    # print(framedistance.max())
    maxframecount.append(framedistance.max())
    for i in range(dfnp.shape[0]):
        distence = dfnp[i, 2] - dfnp[i, 1]
        framecount.append(distence)
        if distence == 0:
            d0.append(distence)
        elif 0 < distence <= 10:
            d10.append(distence)
        elif 10 < distence <= 20:
            d20.append(distence)
        elif 20 < distence <= 30:
            d30.append(distence)
        elif 30 < distence <= 40:
            d40.append(distence)
        elif 40 < distence <= 50:
            d50.append(distence)
        elif 50 < distence <= 60:
            d60.append(distence)
        elif 60 < distence <= 70:
            d70.append(distence)
        elif 70 < distence <= 80:
            d80.append(distence)
        elif 80 < distence <= 90:
            d90.append(distence)
        elif 90 < distence <= 100:
            d100.append(distence)
        elif 100 < distence <= 200:
            d200.append(distence)
        elif 200 < distence <= 300:
            d300.append(distence)
        elif 300 < distence <= 400:
            d400.append(distence)
        elif 400 < distence <= 500:
            d500.append(distence)
        else:
            d600.append(distence)
    sample_count = sample_count + df1.shape[0]

totalmaxframe = max(maxframecount)

print('Sample_num:' + str(sample_count))
print('MaxFrame:' + str(totalmaxframe))
print('averageFrame:' + str((sum(framecount) / len(framecount))))

# plotting
count = [len(d0), len(d10), len(d20), len(d30), len(d40), len(d50), len(d60), len(d70), len(d80), len(d90), len(d100),
         len(d200), len(d300), len(d400), len(d500), len(d600)]

x = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '200', '300', '400', '500', 'over_600']

for i in range(len(x)):
    plt.bar(x[i], count[i], color=plt.get_cmap('Blues')((i + 2) * 15))

for a, b in zip(x, count):  # 柱子上的数字显示
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
plt.title("Frame count")
plt.xlabel("Frame interval")
plt.ylabel("Count")
plt.show()
