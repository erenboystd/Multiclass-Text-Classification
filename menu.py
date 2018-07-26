import os
import time
from json_parser import parser
from predict import predict
from train import func
from test_parser import parse


print("\n MULTICLASS TEXT CLASSIFICATION FOR BANK DOCUMENTS\n")

while True:
    print(" ------------ MENU ------------")

    print(" 1. TRAIN ")

    print(" 2. CLASSIFY ")

    print(" 0. EXIT  ")

    menu_choice = input(" Enter an input: ")

    if menu_choice == '1':

        print(" Enter the path for dataset (ex. format = ./dataset )")

        dataset_path = input(" Enter an input: ")
        print("\n # Processing...")
        parser(dataset_path)
        print("\n # output.csv created\n")
        print(" # Training...")
        x = dataset_path+'/output.csv'
        func(x)
        print("\n # Train finished\n")

        # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json

    elif menu_choice == '2':
        print(" ------------      ------------")
        print(" Choose classification method (default SVC)\n")
        print(" 1. Multinomial Naive Bayes")
        print(" 2. Linear Support Vector Clusters")
        print(" 3. Support Vector Clusters with ratio \n")

        classification_choice = input(" Enter an input: ")


        print(" Enter the path for folder (ex. format = ./dataset )")

        document_path = input(" Enter an input: ")
        parse(document_path)
        print("# Classifying...")
        predict('./test_output.csv',classification_choice)


    elif menu_choice == '0':

        break;
    else:
        print("\n")