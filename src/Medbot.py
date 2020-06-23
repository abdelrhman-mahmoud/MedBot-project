"""
@author: abdo
email : abdo.elsaadny74@gmail.com 
"""

#import libraries
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

import wikipedia
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import random
import db
from fuzzywuzzy import process



# this function to remove stop words from text
# stop words: words that have no meaning in english like [the,and ,...]  
def stemming(text):
    wt=word_tokenize(text)
    ps = PorterStemmer()
    word=[]
    for i in wt:
        x=ps.stem(i)
        word.append(x)
    return word

def stopWords(text):
    #text is a sentence
    englishword = set(stopwords.words('english'))
    filtered = []
    words = word_tokenize(text)
    for i in words:
        if i not in englishword:
            filtered.append(i)
    return filtered

def sorry():
    messages = ["I'm sorry I could not understand that. Let's try again."]
    return messages

#great function 
def ask_symptoms():
    messages = ['Type your symptoms']
    return messages
def greet():
    greeting=['hi','hello']  
    for i in greeting: 
        gr = random.choice (greeting)
    messages = [gr,"I'm MedicalBot, your personal health assistant.\n I can do that for you : \n 1-diagnoses of illnesses. \n 2-Book intensive care unit. \n PLZ select number of service that you want : "]
    return messages
# greet()
'''    
def chosing ():
    messages1="I can do that for you :"
    messages2="1-diagnoses of illnesses."
    messages3="2-Book intensive care unit."
    messages4="PLZ select number of service that you want :"
    return messages1 , messages2  , messages3 , messages4
    '''
    

def asknames():
    askname= ["what's your name ?  ",'your name  ? ']
    for i in askname:
        na=random.choice(askname)
    messages = [na]
    return messages
    
# asknames()

def getName(text):
   
    filtered = stopWords(text)


    tag = nltk.pos_tag(filtered)
   
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
    messages = f"Welcome {name[0]}"
    return messages

# getName("my name is abdelrhman ")



# greet()
# yourname =asknames()
# myname=getName(yourname)

def askAges(uuid):
   askage=['how old are you  ? ',"i'd like to know your age ? ",'tell me your age ? ']
   for i in askage:
       age=random.choice(askage)
       
   messages = [getName(db.get_name(uuid)),age]
   return messages
   #inp = input()
   #return inp
# askAges()
    
def getAge(uuid,inage):

    filtered = stopWords(inage)
    
    for i in filtered:
       filtered = stopWords(inage)
    for i in filtered:
        try:
            age = int(i)
        except Exception:
            continue
    #print(myname,' : ' ,age)
    messages = [str(db.get_name(uuid))+":"+str(age),askGender()]
    return messages

# yourAge = askAges()
# age = getAge(yourAge)

#this function to ask user about his gender
    
def askGender():
    messages = 'Are you a Male or a Female?'
    return messages

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

# -----------------------------------------------------------------------------
class Natural_language_processing:

    def __init__(self):
       
        self.info = []
        
        
        
    def extract (self,text):
        stopWords(text)
        token=stemming(text)

        tagged=nltk.pos_tag(token)
        
        chunkgram=r"""chunk : {<.*>+}
                        		}<VB.?|IN|DT|TO|NNS|CC>+{
                    
                    
                                
                                
                                
                 chunk:
                    {<DT><NN>+<VBG>|<DT><NN|NNS>+}
                    }<DT>{
                    
                    chunk:
                    {<NN><IN><DT>}
                    }<NN>{
                    }<DT>{
                    chunk:
                    {<VB|VBN><RP|IN>}
                    }<VB>{
                    }<VBN>{
                    chunk:
                    {<CD>}
                                
                          chunk:
                    {<WP><VBZ><DT><NN><NN><IN><NNP|NN>+}
                    }<WP>{
                    }<VBZ>{
                    }<DT>{
                    }<IN>{
                    <NN>}{<NN>
        
        
                          chunk:
                    {<JJ>?<NN>+}
                    <JJ>}{<NN>
                    <NN>}{<NN>
                                
                                
                                
                                
                                """
        
        self.info=[]
        chunkparser=nltk.RegexpParser(chunkgram)
        chunked=chunkparser.parse(tagged)
        # chunked.draw() 
    
    
    
        for element in chunked:
            if hasattr(element, 'label'):
                temp = ' '.join(e[0] for e in element)
                self.info.append(temp)
            
        return self.info
    
NLP=Natural_language_processing()
             

# Function to convert the list to string    
def listToString(text):  
    
    str1 = " " 
    
    return (str1.join(text)) 

        
    
# -----------------------------------------------------------------------------
df = pd.read_csv('datasets\diseasedata.csv')
df.isnull().sum().sort_values(ascending=False)
df['prognosis'].value_counts(normalize = True)
df.dtypes.unique()

x = df.drop(['prognosis'],axis =1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_test.shape

DTC = DecisionTreeClassifier(criterion='entropy', max_depth=10,random_state=80)



# knn=KNeighborsClassifier()
# knn.fit(x_train,y_train)
# knn.predict(x_test)
# # print ('scores for train= ',knn.score(x_test, y_test))
# # print('scores for test : ' , knn.score(x_test, y_test))
# y_pred = knn.predict(x_test)
# print('accuracy_score:',accuracy_score(y_pred,y_test))

test_scores={}
train_scores={}
for i in range(2,4,2):
    kf = KFold(n_splits = i)
    sum_train = 0
    sum_test = 0
    data = df
    for train,test in kf.split(data):
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        x_train = train_data.drop(["prognosis"],axis=1)
        y_train = train_data['prognosis']
        x_test = test_data.drop(["prognosis"],axis=1)
        y_test = test_data["prognosis"]
        algo_model = DTC.fit(x_train,y_train)
        sum_train += DTC.score(x_train,y_train)
        y_pred = DTC.predict(x_test)
        sum_test += accuracy_score(y_test,y_pred)
    average_test = sum_test/i
    average_train = sum_train/i
    test_scores[i] = average_test
    train_scores[i] = average_train


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

# -----------------------------------auto correction----------------------------
Replacement_pattern ={"rash skin":"skin_rash","skin rash":"skin_rash",
                    
                       " sneez":"continuous_sneez","continuous sneaz":"continuous_sneezing",
                       
                       "stomach pain" :"stomach_pain","pain stomache":"stomach_pain",
                       "muscle wast":"muscle_wasting","wast muscle":"muscle_wasting",
                       "cold hands":"cold_hands_and_feets","cold hand and feet":"cold_hands_and_feets","cold feets":"cold_hands_and_feets",
                       "weight gain":"weight_gain","gain weight":"weight_gain",
                       "weight loss":"weight_loss","loss weight":"weight_loss",
                       
                       "high fever":"high_fever","fever high":"high_fever",
                       "breathless":"breathlessness","low breath":"breathlessness",
                       "head mild fever":"headache","ashe":"headache",
                       "back pain":"back_pain","pain back":"back_pain",
                       "runny nose":"runny_nose","nose runy":"runny_nose",
                       "chest pain":"chest_pain","pain chest":"chest_pain",
                       "fast heart ":"fast_heart_rate","fast heart rate":"fast_heart_rate",
                        "neck pain":"neck_pain","pain neck":"neck_pain"

    }


def expand_contractions(sentence, text): 
     
    contractions_pattern = re.compile('({})'.format('|'.join(text.keys())),  
                                      flags=re.IGNORECASE|re.DOTALL) 
    def expand_match(contraction): 
        match = contraction.group(0) 
        first_char = match[0] 
        expanded_contraction = text.get(match) if text.get(match) else text.get(match.lower())                        
        expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction 
         
    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
    return expanded_sentence 
 

def splitting(text): 
    return ([i for item in text for i in item.split()]) 

with open("datasets\sym.txt","r") as f :
    sym=f.read().split('\n')

def match(query,choise,limit=1):
    result=process.extract(query,choise,limit=limit)
    return result



def fuzzy(text):
    text1=word_tokenize(text)
    correction=[]
    for i in text1:
        a=match(i,sym)
        if (a[0][1]<80):
            continue
        else:
            correction.append(a[0][0])
    return correction
# ---------------------------------end autocorrection-------------------------------


def getdisease(symptoms):
    
    l=NLP.extract(symptoms)
    lts=listToString(l)
    # print (lts)
    
    expanded_corpus =[expand_contractions(txt, Replacement_pattern)  
                     for txt in sent_tokenize(lts)]
    words=splitting(expanded_corpus)
    lts1=listToString(words)
    # print(lts1)
    correction_word=fuzzy(lts1)
    # print(correction_word)
      
    token = [str(x) for x in correction_word]
    a=[]
    compare=[item for item in token if item in x.columns]
      
    for i in (x.columns):
        
        if i in compare:
            a.append(1)
        elif i not in compare:
            a.append(0)
        else:
            return sorry()
            
    y_diagnosis = DTC.predict([a])
    y_pred_2 = DTC.predict_proba([a])
    # prediction = f"i predict you have {y_diagnosis[0]} disease, confidence score of : {y_pred_2.max()* 100}%"
    # prediction = (('i predict you have  %s  disease, confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
        
    wiki=str(y_diagnosis[0])
    #print ('this is info about your disease :')
    #print ('\n',wikipedia.summary(wiki, sentences=2))
    
    #print(('Name = %s , Age : = %s') %(i_name,i_age))

    messages = [f"i predict you have {y_diagnosis[0]} disease, confidence score of : {y_pred_2.max()* 100}%",'this is info about your disease :',wikipedia.summary(wiki, sentences=2),'note : \n Do not depend on this result .. Please see a doctor']
    return messages

#def note():
#    messages=['note : \n Do not depend on this result .. Please see a doctor']
#    return messages

# getdisease('cough and brethless and high fever')

            
    

#start conversation :the point that acully start with 

"""
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
"""



if __name__=='__main__':
    print (greet())
    inpp=str (input())
    if inpp=='1':
        print(asknames())
        names=input()
        print (getName(names))

# askAges()
# ages=input()
# getAge()
    
        print (askGender())
        genders=input()
        getGender(genders)
        while genders==0:
            print(sorry())
            print (askGender())
            genders=input()
            print(getGender(genders))
            
        print(ask_symptoms())
        symptoms=str (input())
        print (getdisease(symptoms))
            
       
    elif inpp=='2':
        print ('This service will be available in the near future')
    
        
        
        
# getdisease(' high fever and breathless and cough')
        
        
        
