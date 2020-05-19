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




# this function to remove stop words from text
# stop words: words that have no meaning in english like [the,and ,...]  
def stopWords(text):
    #text is a sentence
    a = set(stopwords.words('english'))
    filtered = []
    words = word_tokenize(text)
    for i in words:
        if i not in a:
            filtered.append(i)
    return filtered
#this function to return the stem word from text
def stemming(text):
    #text could be a sent or word
    ps = PorterStemmer()
    empty = []
    for w in text:
        empty.append(w)
    return empty

def sorry():
    print("I'm sorry I could not understand that. Let's try again.")

#great function 

def greet():
    greeting=['hi','hello']
    
    for i in greeting:
        
        gr = random.choice(greeting)
    print('Medbot :' , gr)

# greet()

def asknames():
    askname= ["what's your name ?  ",'your name  ? ']
    for i in askname:
        na=random.choice(askname)
    print ('Medbot : ' , na)
    inp = input()
    return inp
    
# asknames()

def getName(text):
   
    filtered = stopWords(text)
    stemmed = stemming(filtered)

    tag = nltk.pos_tag(stemmed)
   
    noun=[]
    for i in range(len(tag)):

        if ((str(tag[i][1])=='NN' or str(tag[i][1])=='NNP') and str(tag[i][0])!='name'):
            noun.append(tag[i][0])

    chunkGram = r"""Chunk: {<NN+>*} 
                    }<VB>{
                    }<DT>{
                    }<IN>{
                    }<VBD>{
                    }<JJ>{
                    }<NN>{
    
    """
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tag)

    for i in chunked:
        if i != ('name','NN','VB','DT','IN','VBD','JJ'):
            name = i

    print('Medbot : ' ,'welcome',name[0])
    return name[0]



# greet()
# yourname =asknames()
# myname=getName(yourname)

def askAges():
   askage=['how old are you  ? ',"i'd like to know your age ? ",'tell me your age ? ']
   for i in askage:
       age=random.choice(askage)
       
   print ('Medbot : ' , age)
   inp = input()
   return inp
       
#askAge()
    
def getAge(str):

    filtered = stopWords(str)
    
    for i in filtered:
       filtered = stopWords(str)
    for i in filtered:
        try:
            age = int(i)
        except Exception:
            continue
    print(myname,' : ' ,age)

# yourAge = askAges()
# age = getAge(yourAge)

#this function to ask user about his gender
    
def askGender():
    print('Are you a Male or a Female?')
    inp = input()
    return inp

# this function to return gender of user  
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

# askGender()
# getGender()
        
    

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


# df['prognosis'].value_counts(normalize = False).plot.scatter()
# plt.subplots_adjust(left = 0.9, right = 2 , top = 2, bottom = 1)


def getdisease():
    userinputs=str(input('type your symptoms : '))
    
    token = [str(x) for x in userinputs.split()]
    # print (token)
    a=[]
    compare=[item for item in token if item in x.columns]
      
    for i in (x.columns):
        
        if i in compare:
            a.append(1)
        elif i not in compare:
            a.append(0)
        else:
            sorry()
            getdisease()
            
        
    y_diagnosis = knn.predict([a])
    y_pred_2 = knn.predict_proba([a])
    print(('i predict you have  %s  disease, confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
    
    print ('--------------------------------------------------------------')
    
    wiki=str(y_diagnosis[0])
    print ('this is info about your disease :')
    print ('\n',wikipedia.summary(wiki, sentences=2))
    
    #print(('Name = %s , Age : = %s') %(i_name,i_age))
    
    print (' \n\n note : \n Do not depend on this result .. Please see a doctor ')

    return



            
    

#start conversation :the point that acully start with 
greet()
print("I'm MedicalBot, your personal health assistant.")
print("I can do that for you : \n 1-diagnoses of illnesses. \n 2-Book intensive care unit. ")
inpp=int(input('PLZ select number of service that you want : '))

if inpp == 1:
    yourname = asknames()
    myname=getName(yourname)

    yourAge = askAges()
    age = getAge(yourAge)
    
    yourgender = askGender()
    gender = getGender(yourgender)
    while gender==0:
        sorry()
        ufGender = askGender()
        gender = getGender(ufGender)
 
    getdisease()       
    
elif inpp==2:
     print ('you are choosed 2')




















    
    

