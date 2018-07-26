import json
import os
import sys
import glob
import re

def parser(data_path):
    csv_output = data_path + '/output.csv'
    output = open(csv_output,'w',encoding="utf-8")
    output.write("sinif\ticerik\n")

    path = data_path+'/EFT'
    for i in glob.glob(os.path.join(path, '*.json')):
        with open(i,encoding = "utf-8") as f:
            data = json.load(f)


        output.write("""EFT\t""")
        for i in range (0,len(data['regions'])):
            for j in range(0,len(data['regions'][i]['lines'])):

                for k in range(0,len(data['regions'][i]['lines'][j]['words'])):
                    string = data['regions'][i]['lines'][j]['words'][k]['text']
                    string = re.sub(r'\b[0-9]+\b\s*', '', string)
                    output.write(string)
                    if check_words(data,i,j,k) != 0:
                        output.write(" ")
        output.write("\n")

    path = data_path+'/HAVALE'
    for i in glob.glob(os.path.join(path, '*.json')):
        with open(i,encoding = "utf-8") as f:
            data = json.load(f)

        output.write("""HAVALE\t""")
        for i in range (0,len(data['regions'])):
            for j in range(0,len(data['regions'][i]['lines'])):
                for k in range(0,len(data['regions'][i]['lines'][j]['words'])):
                    string = data['regions'][i]['lines'][j]['words'][k]['text']
                    string = re.sub(r'\b[0-9]+\b\s*', '', string)
                    output.write(string)
                    if check_words(data,i,j,k) != 0:
                        output.write(" ")

        output.write("\n")


    path = data_path+'/VIRMAN'
    for i in glob.glob(os.path.join(path, '*.json')):
        with open(i,encoding = "utf-8") as f:
            data = json.load(f)

        output.write("""VIRMAN\t""")
        for i in range (0,len(data['regions'])):
            for j in range(0,len(data['regions'][i]['lines'])):
                for k in range(0,len(data['regions'][i]['lines'][j]['words'])):
                    string = data['regions'][i]['lines'][j]['words'][k]['text']
                    string = re.sub(r'\b[0-9]+\b\s*', '', string)
                    output.write(string)
                    if check_words(data,i,j,k) != 0:
                        output.write(" ")
        output.write("\n")


    path = data_path+'/OTHER'
    for i in glob.glob(os.path.join(path, '*.json')):
        with open(i,encoding = "utf-8") as f:
            data = json.load(f)

        output.write("""OTHER\t""")
        for i in range (0,len(data['regions'])):
            for j in range(0,len(data['regions'][i]['lines'])):
                for k in range(0,len(data['regions'][i]['lines'][j]['words'])):
                    string = data['regions'][i]['lines'][j]['words'][k]['text']
                    string = re.sub(r'\b[0-9]+\b\s*', '', string)
                    output.write(string)
                    if check_words(data,i,j,k) != 0:
                        output.write(" ")
        output.write("\n")

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

