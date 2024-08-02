import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np 
import csv
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")


training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Grouping data based on 'prognosis' column
reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)


model = SVC()
model.fit(x_train, y_train)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}


for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)

def convert_to_lowercase(text):
    return text.lower()

def check_pattern(dis_list, inp, chk_dis):
    inp_lower = inp.lower()  # Convert input to lowercase
    pred_list = []
    inp_lower = inp_lower.replace(' ', '_')
    patt = f"{inp_lower}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item.lower())]  # Convert items in dis_list to lowercase for comparison
    print("\nEnter the symptoms you are experiencing (separated by commas): ")
    symptoms_input = input("").strip().split(',')
    symptoms_input = [symptom.strip().lower() for symptom in symptoms_input]  # Convert to lowercase

    # Check if entered symptoms are valid
    invalid_symptoms = [symptom for symptom in symptoms_input if symptom not in chk_dis]
    if invalid_symptoms:
        print(f"The following symptoms are not valid: {', '.join(invalid_symptoms)}. Please try again.")
        return

    # Convert items in dis_list to lowercase for comparison
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []
    
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    print("\nEnter the symptoms you are experiencing (separated by commas): ")
    symptoms_input = input("").strip().split(',')
    symptoms_input = [symptom.strip() for symptom in symptoms_input]

    # Check if all symptoms are valid
    invalid_symptoms = [symptom for symptom in symptoms_input if symptom not in chk_dis]
    if invalid_symptoms:
        print(f"The following symptoms are not valid: {', '.join(invalid_symptoms)}. Please try again.")
        return

    while True:
        try:
            num_days = int(input("Okay. From how many days ? : "))
            break
        except ValueError:
            print("Please enter a valid number of days.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name in symptoms_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                if syms not in symptoms_input:  # Check if symptom has not already been specified
                    inp = ""
                    print(syms, "? : ", end='')
                    while True:
                        inp = input("")
                        if inp == "yes" or inp == "no":
                            break
                        else:
                            print("Provide proper answers i.e. (yes/no) : ", end="")
                    if inp == "yes":
                        symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])
            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precaution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precaution_list):
                print(i + 1, ")", j)

    recurse(0, 1)


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index,
                     symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
print("----------------------------------------------------------------------------------------")
