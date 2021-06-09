import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_excel('C:/Users/siddh/Downloads/Data_Train.xlsx')

train_data.dropna(inplace=True)
train_data.drop('Additional_Info',inplace=True,axis=1)
train_data.drop('Route',inplace=True,axis=1)
airline=train_data['Airline']
airline_hot=pd.get_dummies(airline,drop_first=True)
source=train_data['Source']
source_hot=pd.get_dummies(source,prefix='source')

destination=train_data['Destination']
destination_hot=pd.get_dummies(destination,prefix='destination')
train_data['journey_day']=pd.to_datetime(train_data['Date_of_Journey']).dt.day

train_data['journey_month']=pd.to_datetime(train_data['Date_of_Journey']).dt.month
train_data.drop('Date_of_Journey',inplace=True,axis=1)

train_data.drop('journey_date',inplace=True,axis=1)
train_data.drop('Airline',inplace=True,axis=1)

train_data.drop('Source',inplace=True,axis=1)

train_data.drop('Destination',inplace=True,axis=1)

dep_time=train_data['Dep_Time']
train_data['dep_hour']=pd.to_datetime(dep_time).dt.hour
train_data['dep_minute']=pd.to_datetime(dep_time).dt.minute
train_data.drop('Dep_Time',axis=1,inplace=True)

arr_time=train_data['Arrival_Time']
train_data['arr_hour']=pd.to_datetime(arr_time).dt.hour
train_data['arr_minute']=pd.to_datetime(arr_time).dt.minute

train_data.drop('Arrival_Time',axis=1,inplace=True)
stops={'Total_Stops':{
    '1 stop':1,
    'non-stop':0,
    '2 stops':2,
    '3 stops':3,
    '4 stops':4
}}
train_data.replace(stops,inplace=True)
duration=list(train_data['Duration'])
duration_min=[]
for i in range(len(duration)):
  duration_split=duration[i].split(' ')
  if(len(duration_split)==1):
    if 'h' in duration_split[0]:
      duration[i]=duration[i]+' 0m'
    if 'm' in duration_split[0]:
      duration[i]='0h '+duration[i]

for i in range(len(duration)):
  duration_split=duration[i].split(' ')
  hr=duration_split[0].replace('h','')
  hr=int(hr)
  min=duration_split[1].replace('m','')
  min=int(min)
  duration_min.append(min+(hr*60))


train_data['Duration']=duration_min

data=pd.concat([train_data,airline_hot,source_hot,destination_hot],axis=1)

price=data['Price']
data.drop('Price',axis=1,inplace=True)

features=data.iloc[:,:]
features=np.array(features)
price=np.array(price)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,price,test_size=0.2,random_state=7)
from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(x_train,y_train)

######################################################################################
#   Model Selection ##
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(max_depth=8,n_estimators=250)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc= model.score(x_train,y_train)
print(acc)

#########################################################################
import pickle
pickle.dump(model,open('flight_rf.pkl','wb+'))