import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

#Loading Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Cleaning Data
print(train.shape) #Total data available
print(train.isna().sum()) #NaN values
train_dropped = train.dropna(subset=['keyword'], axis=0)
replace = train_dropped.location.value_counts().idxmax()
train_cleaned = train_dropped.fillna(value=replace) #Filling missing location values with most frequent location
print(train_cleaned.isna().sum()) #All missing values filled
print(train_cleaned.shape) #Data available after cleaning

# Visualization of training data
train_cleaned.groupby('target').id.count().plot.bar(ylim=0)
plt.show()
item_counts = train_cleaned["target"].value_counts()
print(item_counts)
baseline_accuracy = 4323/7552
print(baseline_accuracy)

#Lowercase textual training data
train_cleaned['location'] = train_cleaned["location"].str.lower()
train_cleaned['text'] = train_cleaned["text"].str.lower()

#Remove Punctuations
def remove_punctuation_marks(x):
    try:
        x = x.str.replace('[^\w\s]','')
    except:
        pass
    return x
train_cleaned = train_cleaned.apply(remove_punctuation_marks)

#Setting features and targets
y = train_cleaned[['target']]
X = train_cleaned.drop(axis=1, columns=['target'])
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)

#Vectorizing Data
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words = 'english', ngram_range=(1,3))

X_train_v = vectorizer.fit_transform(X_train.text)
X_valid_v = vectorizer.transform(X_valid.text)

#Model training
model = XGBClassifier(use_label_encoder=False)

model.fit(X_train_v,y_train)
y_preds = model.predict(X_valid_v)
y_preds = y_preds.reshape(1511,1)
y_preds = pd.DataFrame(y_preds, columns=['target']).astype('float')

f1_score = f1_score(y_preds, y_valid)
print(f1_score)

#Lowercase textual test data
test['location'] = test["location"].str.lower()
test['text'] = test["text"].str.lower()

#Remove Punctuations
test = test.apply(remove_punctuation_marks)

#Vectorizing Data
X_test_v = vectorizer.transform(test.text)

#Predictions and exporting
submission = model.predict(X_test_v)
submission = np.reshape(submission, (3263,1))
submission_csv = pd.DataFrame(test.id, columns = ['id'])
submission_csv['target'] = pd.DataFrame(submission, columns=['target']).astype('int')
submission_csv1 = submission_csv.set_index('id')

submission_csv1.to_csv('submission.csv')