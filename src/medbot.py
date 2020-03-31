#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import wikipedia
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import random


#Greeting datasets 
gr=pd.read_csv('C:\\Users\\M\\Desktop\\my project\\Greeting.csv',engine='python')
gr = np.array(gr)
gg = gr[:,0]
#print (gr)

#welcome datasets
w = pd.read_csv('C:\\Users\\M\\Desktop\\my project\\welcome.csv', engine='python')
w = np.array(w)
ww = w[:,0]
#print (wc)


#age datasets
ag = pd.read_csv('C:\\Users\\M\\Desktop\\my project\\AGE.csv', engine='python')
ag = np.array(ag)
age = ag[:,0]
#print (ag)


#bye datasets
by = pd.read_csv('C:\\Users\\M\\Desktop\\my project\\BYE.csv', engine='python')
by = np.array(by)
bye = by[:,0]
#print (by)


#name datasets
nm = pd.read_csv('C:\\Users\\M\\Desktop\\my project\\Name.csv', engine='python')
nm = np.array(nm)
nn = nm[:,0]
#print (nm)

#Diseases datasets
sr = pd.read_csv('C:\\Users\\M\\Desktop\\my project\\sym.csv', engine='python')
sr = np.array(sr)
dis = sr[:,1]
symp = sr[:,2]

dis = np.array(dis)
symp = np.array(symp)
#print (dis)
#print(symp)


def stopWords(text):
    #text is a sentence
    a = set(stopwords.words('english'))
    filtered = []
    words = word_tokenize(text)
    for i in words:
        if i not in a:
            filtered.append(i)
    return filtered

def stemming(text):
    #text could be a sent or word
    ps = PorterStemmer()
    empty = []
    for w in text:
        empty.append(w)
    return empty

def greet():
    a = random.randint(0,20)
    print(gr[a%5])
#greet()



def askName():
    a = random.randint(0,20)
    print(nn[a%2])
    inp = input()
    return inp
#askName()
def getName(text):
   
    filtered = stopWords(text)
    stemmed = stemming(filtered)
##    print("stemmed",stemmed)
    tag = nltk.pos_tag(stemmed)
    #print(tag)
    noun=[]
    for i in range(len(tag)):
##        print(tag[i][1])
        if ((str(tag[i][1])=='NN' or str(tag[i][1])=='NNP') and str(tag[i][0])!='name'):
            noun.append(tag[i][0])
##    print(noun)
    chunkGram = r"""Chunk: {<NN+>*}  """
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tag)
#    print(chunked)
    for i in chunked:
        if i != ('name','NN'):
            name = i

    print('welcome',name[0])
    return name
def askAge():
    a = random.randint(0,30)
    print(age[a%7])
    inp = input()
    return inp
#askAge()
    


def getAge(text):

    filtered = stopWords(text)
    
    for i in filtered:
       filtered = stopWords(text)
    for i in filtered:
        try:
            age = int(i)
        except Exception:
            continue
    return age
            

def askGender():
    print('Are you a Male or a Female?')
    inp = input()
    return inp


    
    
def getGender(text):

    filtered = stopWords(text)
    flag=0
    for i in filtered:
        if i.lower()=='male' or i.lower()=='female':
            
            gender = i
            flag=1
    if flag!=1:
        return 0
    else:
        return gender    

def sorry():
    print("I'm sorry I could not understand that. Let's try again.")
df = pd.read_csv('C:\\Users\\M\Desktop\\my project\\Training.csv')
df.isnull().sum().sort_values(ascending=False)
df['prognosis'].value_counts(normalize = True)
df.dtypes.unique()

x = df.drop(['prognosis'],axis =1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_test.shape


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.predict(x_test)
# print ('scores for train= ',knn.score(x_test, y_test))
# print('scores for test : ' , knn.score(x_test, y_test))
y_pred = knn.predict(x_test)
# print('accuracy_score:',accuracy_score(y_pred,y_test))



#for check only
# real_diseases = y_test.values

# for i in range(0, len(real_diseases)):
#     if y_pred[i] == real_diseases[i]:
#         # print ('Pred: {0} Actual:{1}'.format(y_pred[i], real_diseases[i]))
#     else:
#         # print('worng prediction')
#         # print ('Pred: {0} Actual:{1}'.format(y_pred[i], real_diseases[i]))
        
        
        
CM = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix is : \n', CM)
        
# sns.heatmap(CM, center = True)

# plt.show()        
yourName = askName()
name= getName(yourName)

    
yourAge = askAge()
age = getAge(yourAge)


yourGender = askGender()
gender = getGender(yourGender)
while gender==0:
    sorry()
    yourGender = askGender()
    gender = getGender(yourGender)

print('your name is {} and your age is {} and your gender is {}'.format(name[0],age,gender))        
a = list(range(2,134))

for i in range(len(x.columns)):
    print(str(i+1+1) + ":", x.columns[i])
choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
b = [int(x) for x in choices.split()]
count = 0
while count < len(b):
    item_to_replace =  b[count]
    replacement_value = 1
    indices_to_replace = [i for i,x in enumerate(a) if x==item_to_replace]
    count += 1
    for i in indices_to_replace:
        a[i] = replacement_value
a = [0 if x !=1 else x for x in a]
y_diagnosis = knn.predict([a])
y_pred_2 = knn.predict_proba([a])
print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
print ('--------------------------------------------------------------')

wiki=str(y_diagnosis[0])
print ('this is info about your disease :')
print ('\n',wikipedia.summary(wiki, sentences=2))

#print(('Name = %s , Age : = %s') %(i_name,i_age))
        
        
        
        