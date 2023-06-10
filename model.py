import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle 

df = pd.read_csv("insurance.csv")

le= preprocessing.LabelEncoder()

df["sex"]=le.fit_transform(df["sex"])

df["smoker"]=le.fit_transform(df["smoker"])

df["region"]=le.fit_transform(df["region"])
 
df.head()

#statsmodel 
import statsmodels.formula.api as smf

model=smf.ols("charges ~ age+sex+bmi+children+smoker+region",data=df).fit()

model.params

# Select independent and dependent variable
y=df["charges"]
x=df.drop(["charges"],axis=1)

# Split the dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# Instantiate the model
regressor = RandomForestRegressor()

# Fit the model
regressor.fit(x_train, y_train)

# Make pickle file of our model
pickle.dump(regressor, open("model.pkl", "wb"))