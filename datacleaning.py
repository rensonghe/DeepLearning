#-*-coding:utf-8 -*-

import pandas as pd

# df = pd.read_csv('Z:/TickFile/ni/20150327.txt',delimiter="\t")
import csv
with open('20150327.csv', 'w+', newline='') as csvfile:
    spamriter = csv.writer(csvfile, dialect='excel')
    with open('Z:/TickFile/ni/20150327.txt', 'r',encoding='utf-8') as filein:
        for line in filein:
            line_list = line.strip('\n').split(',') # 因为我的txt文件是“，”分割
            spamriter.writerow(line_list)



