import pandas as pd
import numpy as np
import sys
import pickle
import time
from io import StringIO
import matplotlib.pyplot as plt
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics  import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from iban_check import check_valid_iban


def predict(file,choice):

    pikl = pickle.load(open('./MultiNB.pikl','rb'))
    pik_v = pickle.load(open('./unigram_vocab.pikl','rb'))
    pikl_linear_SVC = pickle.load(open('./Linear_SVC.pikl','rb'))
    pikl_SVC = pickle.load(open('./SVC.pikl','rb'))
    count_vect = CountVectorizer(vocabulary = pik_v)
    ot = open('./Predict_Results.txt','wt')

    text = pd.read_csv(file, sep = '\t')

    tot_eft = 0
    tot_other = 0
    tot_havale = 0
    tot_virman = 0
    i = 0

    temp = []
    temp_string = []

    while text.iloc[i][0] != "\EOF":
        #print(text.iloc[i][0])        
        
        string = "" + text.iloc[i][0]

        print("Processing...")

        ot.write('\n\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n')
        ot.write('TEST FILE SOURCE : ')
        print(text.iloc[i][1])
        ot.write(text.iloc[i][1])


        strtolist = list(string)

        iban = iban_find(strtolist)
        if check_digit(iban) == 1:
            new_iban = ''.join(iban)
            ot.write("\nIBAN NO :")
            if check_valid_iban(new_iban,ot) == 1:
                ot.write(" # Valid iban \n")
            else:
                ot.write(" # Invalid iban \n")

        ot.write("\nPREDICT : ")

        if choice == '1':
            rslt = pikl.predict(count_vect.transform([string]))
            rslt = ''.join(rslt)
            ot.write(rslt)
            if(rslt == 'EFT'):
                tot_eft+=1
            elif(rslt == 'HAVALE'):
                tot_havale+=1
            elif(rslt == 'OTHER'):
                tot_other+=1
            elif(rslt == 'VIRMAN'):
                tot_virman+=1
        elif choice == '2':
            rslt = pikl_linear_SVC.predict(count_vect.transform([string]))
            rslt = ''.join(rslt)
            ot.write(rslt)
            if(rslt == 'EFT'):
                tot_eft+=1
            elif(rslt == 'HAVALE'):
                tot_havale+=1
            elif(rslt == 'OTHER'):
                tot_other+=1
            elif(rslt == 'VIRMAN'):
                tot_virman+=1
        elif choice == '3':
            proba = pikl_SVC.predict_proba(count_vect.transform([string]))
            classes = pikl_SVC.classes_
            print_max_proba(proba,classes,ot)
        print('\n\n')
        i += 1


    if choice != '3':
        ot.write("\n - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        ot.write("\n EFT ratio : ")
        ot.write(str(tot_eft/(i)))
        ot.write("\n HAVALE ratio : ")
        ot.write(str(tot_havale/(i)))
        ot.write("\n VIRMAN ratio : ")
        ot.write(str(tot_virman/(i)))
        ot.write("\n OTHER ratio : ")
        ot.write(str(tot_other/(i)))
        ot.write("\n Total document count : ")
        ot.write(str(i))

def iban_find(temp):
    iban = []
    for i in range(0,len(temp)-1):
        if temp[i] == 'T' and temp[i+1] == 'R':
            j=0
            while j<26:
                if i < len(temp):
                    if temp[i] == 'E' or temp[i] == 'B':
                        temp[i] = '8'
                    elif temp[i] == 'S':
                        temp[i] = '5'
                    elif temp[i] == 'O':
                        temp[i] = '0'
                    if temp[i] != ' ':
                        iban.append(temp[i])
                        j+=1
                elif i > len(temp):
                    break
                i+=1
    return iban

def check_digit(iban):
    i = 0
    while i < len(iban):
        if i != 0 and i != 1:
            if iban[i].isdigit() == 0:
                return 0
        i += 1
    return 1

def print_max_proba(proba,classes,ot):

    str_proba = proba.tolist()
    str_proba = str(str_proba)

    proba = str_proba
    str_proba = str_proba.split(',')

    proba = str_proba

    first = proba[0]
    last = proba[3]


    first = list(first)
    first = first[4:]
    first.insert(2,'.')
    first = ''.join(first)
    first = float(first)

    last = list(last)
    last = last[:-2]
    last = last[3:]
    last.insert(2,'.')
    last = ''.join(last)
    last = float(last)

    one = proba[1]
    one = list(one)
    one = one[3:]
    one.insert(2,'.')
    one = ''.join(one)
    one = float(one)

    two = proba[2]
    two = list(two)
    two = two[3:]
    two.insert(2,'.')
    two = ''.join(two)
    two = float(two)

    if first > one:
        select_one = first
    else:
        select_one = one
    if two > last:
        select_two = two
    else:
        select_two = last

    if select_one > select_two:
        max = select_one
    else:
        max = select_two

    if  max == first:
        ot.write(' %')
        ot.write(str(first))
        ot.write('\n')
        ot.write(classes[0])
    elif max == one:
        ot.write(' %')
        ot.write(str(one))
        ot.write('\n')
        ot.write(classes[1])
    elif max == two:
        ot.write(' %')
        ot.write(str(two))
        ot.write('\n')
        ot.write(classes[2])
    elif max == last:
        ot.write(' %')
        ot.write(str(last))
        ot.write('\n')
        ot.write(classes[3])