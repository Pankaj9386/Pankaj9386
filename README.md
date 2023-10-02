#removing stopwords from each review
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[] #text collection

for i in range(0,1000):
   review =re.sub(pattern='[^a-zA-Z]',repl=' ', string=data['Review'][i])

   review = review.lower() #covert all upper characters to lower
   review_words = review.split()
   review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

   ps= PorterStemmer() #normalize all words having same meaning
   review =[ps.stem(word) for word in review_words]

   review = ' '.join(review)
   corpus.append(review) #append to collection
   
corpus[:1500] #all reviews after removing

from sklearn.feature_extraction.text import CountVectorizer
countv = CountVectorizer(max_features=1500)
X =countv.fit_transform(corpus).toarray()
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.20, random_state=0)

X_train.shape,X_test.shape, y_train.shape,y_test.shape

#applying bayes algoithm
from sklearn.naive_bayes import MultinomialNB

classifier =MultinomialNB()
classifier.fit(X_train, y_train)

y_prediction = classifier.predict(X_test)

y_prediction

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

score1 =accuracy_score(y_test,y_prediction)
score2 = accuracy_score(y_test,y_prediction)
score3 = recall_score(y_test,y_prediction)

print("WE GOT FOLLOWING RESULT")
print("Accuracy score = {}%".format(round(score1*100,3)))
print("Precision score = {}%".format(round(score2*100,3)))
print("recall score = {}%".format(round(score3*100,3)))

from sklearn.metrics import confusion_matrix
confum = confusion_matrix(y_test, y_prediction)

confum

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize =(20,6))
sns.heatmap(confum, annot=True, cmap="YlGnBu", xticklabels=['NEGATIVE','POSITIVE'],yticklabels=['NEGATIVE','POSITIVE'])
plt.xlabel('VALUES ( PREDICTION)')
plt.ylabel('ACTUAL VALUES')

from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
best_accuracy =0.0
alpha_val =0.0
for i in np.arange(0.1,1.1,0.1):
  legit_classifier =MultinomialNB(alpha=i)
  legit_classifier.fit(X_train,y_train)
  legit_y_pred =legit_classifier.predict(X_test)
  adv_score = accuracy_score(y_test,legit_y_pred)
  print("Accuracy Score for alpha={} is {}%".format(round(i,1),round(adv_score*100,3)))
  if adv_score>best_accuracy:
     best_accuracy=adv_score
     alpha_val =i
print('                                                                    ')
print("The Best Accuracy Score is {}% as alpha value is {}".format(round(best_accuracy*100, 2), round(alpha_val, 1)))

classifier =MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)

    temp = cv.transform([final_review]).toarray()
    return classifier.predict(temp)

sample_review ='The food is good.'

if predict_sentiment(sample_review):
  print("Positive review")

else:
  print("Negative review")

sample_review ='dish is too much oily.'

if predict_sentiment(sample_review):
  print(" Positive review")

else:
  print("Negative review")

sample_review ='very poor service.'

if predict_sentiment(sample_review):
  print("Positive review")

else:
  print("Negative review")

sample_review ='staff behaves very nice.'

if predict_sentiment(sample_review):
  print("Positive review")

else:
  print("Negative review")

sample_review ='no ventilation here.'

if predict_sentiment(sample_review):
  print("Positive review")

else:
  print("Negative review")

sample_review ='i will come to party here.'

if predict_sentiment(sample_review):
  print("Positive review")

else:
  print("Negative review")

