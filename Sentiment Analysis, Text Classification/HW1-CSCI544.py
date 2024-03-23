#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install contractions


# In[2]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

 


# In[ ]:





# ## Read Data

# In[3]:


df= pd.read_table('amazon_reviews_us_Beauty_v1_00.tsv',on_bad_lines='skip',low_memory=False)


# ## Keep Reviews and Ratings

# In[4]:


df_fnl= df[['review_body','star_rating']]


# In[5]:


#df_fnl


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[6]:


df_final=df_fnl.replace({'star_rating':{2:1,3:2,4:3,5:3,'5':3,'2':1,'3':2,'4':3,'1':1}})


# In[7]:


#df_final.isnull().sum()


# In[8]:


df_final.dropna(inplace=True)


# In[9]:


s0 = df_final[df_final['star_rating'].eq(1)].sample(20000).index
s1 = df_final[df_final['star_rating'].eq(2)].sample(20000).index 
s2 = df_final[df_final['star_rating'].eq(3)].sample(20000).index 


# In[10]:


df_fi = df_final.loc[s0.union(s1).union(s2)]


# In[11]:


#df_fi


# In[12]:


#df_fi.isnull().sum()


# # Data Cleaning
# 
# 

# # Pre-processing

# In[13]:


dt_before_clean=df_fi['review_body'].apply(len).mean()


# In[15]:


def remove_alphanumeric(s):
    s=s.lower()
    s=s.strip()
    s=contractions.fix(s)
    s=s.replace(r'<[^<>]*>', '')
    s=s.replace(r'http\S+', '').replace(r'www\S+', '')
    s=re.sub(r'[^a-zA-Z]',' ',s)
    return re.sub(' +', ' ',s)


# In[16]:



df_fi['review_body']=df_fi['review_body'].apply(remove_alphanumeric)


# In[17]:


dt_after_clean=df_fi['review_body'].apply(len).mean()


# In[18]:


print("Average length of reviews before and after data cleaning:",dt_before_clean,',',dt_after_clean)


# In[ ]:





# ## remove the stop words 

# In[19]:


dt_before_preprocess=df_fi['review_body'].apply(len).mean()


# In[20]:


# from nltk.corpus import stopwords
# stop = stopwords.words('english')
# df_fi['review_body'] = df_fi['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
 


# In[21]:


#df_fi


# ## perform lemmatization  

# In[22]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word,pos='v') for word in words]
        return ' '.join(words)

df_fi['review_body']=df_fi['review_body'].apply(lemmatize_words)


# In[23]:


#df_fi


# In[24]:


dt_after_preprocess=df_fi['review_body'].apply(len).mean()


# In[25]:


print("Average length of reviews before and after data preprocessing:",dt_before_preprocess,',',dt_after_preprocess)


# # TF-IDF Feature Extraction

# In[26]:


imp_features = TfidfVectorizer(ngram_range=(1,3))
x = imp_features.fit_transform(df_fi['review_body'])


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(x, df_fi['star_rating'], test_size=0.2, random_state=42)
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# # Perceptron

# In[28]:


from sklearn.linear_model import Perceptron
clf_percep=Perceptron(tol=1e-3, random_state=0)
clf_percep.fit(X_train,Y_train)


# In[29]:


predicted_perceptron = clf_percep.predict(X_test)


# In[30]:


target_names = ['class 1', 'class 2', 'class 3']
#print(classification_report(Y_test, predicted_perceptron, target_names=target_names))


# In[31]:


precision, recall, fscore, support = score(Y_test, predicted_perceptron)

# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))


# In[32]:


print('----------------------------------Perceptron Classification--------------------------------------------------------------------')
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],',','Recall:', recall[i],',', 'F1 score:',fscore[i],',','average:',((precision[i]+recall[i]+fscore[i])/3))


# # SVM

# In[33]:


from sklearn.svm import LinearSVC
clf_SVM= LinearSVC(random_state=42, tol=1e-5)
clf_SVM.fit(X_train,Y_train)


# In[34]:


predicted_SVM=clf_SVM.predict(X_test)


# In[35]:


target_names = ['class 1', 'class 2', 'class 3']
#print(classification_report(Y_test, predicted_SVM, target_names=target_names))


# In[36]:


precision, recall, fscore, support = score(Y_test, predicted_SVM)

#print('precision: {}'.format(precision))
#print('recall: {}'.format(recall))
#print('fscore: {}'.format(fscore))
#print('support: {}'.format(support))


# In[37]:


print('----------------------------------Support Vector Machine Classification--------------------------------------------------------')
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],',','Recall:', recall[i],',', 'F1 score:',fscore[i],',','average:',((precision[i]+recall[i]+fscore[i])/3))


# # Logistic Regression

# In[38]:


from sklearn.linear_model import LogisticRegression

clf_Logistic= LogisticRegression(random_state=42,max_iter=1000000)
clf_Logistic.fit(X_train,Y_train)


# In[39]:


predicted_Log=clf_Logistic.predict(X_test)


# In[40]:


target_names = ['class 1', 'class 2', 'class 3']
#print(classification_report(Y_test, predicted_Log, target_names=target_names))


# In[41]:


precision, recall, fscore, support = score(Y_test, predicted_Log)

#print('precision: {}'.format(precision))
#print('recall: {}'.format(recall))
#print('fscore: {}'.format(fscore))
#print('support: {}'.format(support))


# In[42]:


print('--------------------------------------------Logistic Regression Classification-------------------------------------------------')
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],',','Recall:', recall[i],',', 'F1 score:',fscore[i],',','average:',((precision[i]+recall[i]+fscore[i])/3))


# # Naive Bayes

# In[43]:


from sklearn.naive_bayes import MultinomialNB

clf_naive = MultinomialNB()
clf_naive.fit(X_train, Y_train)


# In[44]:


predicted_naive = clf_naive.predict(X_test)


# In[45]:


conf=confusion_matrix(Y_test,predicted_naive)


# In[46]:


target_names = ['class 1', 'class 2', 'class 3']
#print(classification_report(Y_test, predicted_naive, target_names=target_names))


# In[ ]:





# In[47]:


precision, recall, fscore, support = score(Y_test, predicted_naive)

#print('precision: {}'.format(precision))
#print('recall: {}'.format(recall))
#print('fscore: {}'.format(fscore))
#print('support: {}'.format(support))


# In[ ]:





# In[48]:


print('-----------------------------------------------Naive Bayes Classification-----------------------------------------------------')
for i in range(3):
    print('class '+ str(i+1),'Precision:',precision[i],',','Recall:', recall[i],',', 'F1 score:',fscore[i],',','average:',((precision[i]+recall[i]+fscore[i])/3))


# In[ ]:




