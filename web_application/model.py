import numpy as np
import pandas as pd
import xlrd

import pickle

data = pd.read_csv(r'salary_data.csv')

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in [ 'workclass', 'education', 'marital-status',
       'occupation', 'relationship', 'race', 'sex',
         'native-country', 'salary']:
    data[i]=label_encoder.fit_transform(data[i])

x=data.drop(['salary'],axis=1)
y=data['salary']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=42,test_size=0.25)



from sklearn.ensemble import GradientBoostingClassifier
sv = GradientBoostingClassifier()
sv.fit(x_train,y_train)
y_pred = sv.predict(x_test)


pickle.dump(label_encoder,open(r'label.pkl','wb'))
pickle.dump(sv,open(r'salary.pkl','wb'))