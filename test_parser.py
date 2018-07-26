import json
import os
from iban_check import check_valid_iban
import sys
import glob
import re

def parse(file):

    output = open('./test_output.csv','w',encoding="utf-8")
    temp = []
    temp_string = []

    for z in glob.glob(os.path.join(file, '*.json')):
        with open(z,encoding = "utf-8") as f:
            data = json.load(f)
        output.write("ROW ")
        for i in range (0,len(data['regions'])):
            for j in range(0,len(data['regions'][i]['lines'])):
                for k in range(0,len(data['regions'][i]['lines'][j]['words'])):
                    string = data['regions'][i]['lines'][j]['words'][k]['text']
                    output.write(string)
                    temp = ''.join(string)
                    if check_words(data,i,j,k) != 0:
                        output.write(" ")
        output.write("\t")
        output.write(z)
        output.write("\n")

    output.write("\EOF")
    output.close()

def check_words(data, i,j,k):

    limit = len(data['regions'][i]['lines'][j]['words'])-1

    if k < limit:
        str = data['regions'][i]['lines'][j]['words'][k]['boundingBox']
        str_next = data['regions'][i]['lines'][j]['words'][k+1]['boundingBox']
        xl,yt,w,h = str.split(",")
        xl_next,yt_next,w_next,h_next = str_next.split(",")

        xl = int(xl)
        w = int(w)
        xl_next = int(xl_next)

        if xl_next - (xl+w) < 8:
            return 0
        else:
            return 1
    else:
        return 1
