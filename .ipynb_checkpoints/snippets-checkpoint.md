![ML Map](./MLMap.png)
![ML Decision Sheet](./ml_cheat_sheet.png)

# 1. Importing basics 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
import seaborn as sns
%load_ext autoreload
%autoreload 2
```
  

# 2. Scaling Features  

## 2.1 MinMaxScaler  
  
```python
from sklearn.preprocessing import MinMaxScaler
normalizer = MinMaxScaler() 
normalizer.fit(df[['columns']])
df['columns'] = normalizer.transform(df[['columns']])
```

## 2.2 RobustScaler  

```python
from sklearn.preprocessing import RobustScaler

# Instanciate Robust Scaler
r_scaler = RobustScaler()

# Fit scaler to feature
r_scaler.fit(df[['columns']])

# Scale
df['columns'] = r_scaler.transform(df[['columns']])
```

# 3. Encoding

## 3.1 Label Encoder

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['column'])
df['column'] = le.transform(df['column'])
```

## 3.2 OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse = False)
ohe.fit(df[['columns']])
oh = ohe.transform(df[['columns']])

df["column1"],df["column2"],df['column3'] = oh.T
```

# 4. HoldOut  
  
## 4.1 Training Split  

> Define **X**,**y** first
```python
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3) #X, y used here
```  
  
# 5. Model   

## 5.1 Regression  
  
### 5.1.1 LinearRegression
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)  ## => View the model's slope (a)
print(model.intercept_)  ## => View the model's intercept (b)
model.score(X_test, y_test)
```
  
### 5.1.2 KNNRegressor  
  
```python
from sklearn.neighbors import KNeighborsRegressor 
neigh = KNeighborsRegressor(n_neighbors=2)  #fix k value inside *n_neighbors* parameter  
neigh.fit(X_train, y_train)  ## => Require X_train, y_train from Hold Out method
neigh.score(X_test,y_test)  ## => Require X_test, y_test from Hold Out method
neigh.predict(new_X) ## => new_X : vector of feature to predict
```  

## 5.2 Classification 
  
### 5.2.1 LogisticRegression  
```python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)  ## => Require X_train, y_train from Hold Out method
log_model.score(X_train,y_train)  ## => Require X_test, y_test from Hold Out method
log_model.predict(new_X) ## => new_X : vector of feature to predict  
```
  
### 5.2.2 KNNClassifier

```python
from sklearn.neighbors import KNeighborsClassifier
kneigh = KNeighborsClassifier(n_neighbors=10) ## ==> K = 10 neighbors
kneigh.fit(X_train, y_train)  ## => Require X_train, y_train from Hold Out method
kneigh.score(X_test,y_test)  ## => Require X_test, y_test from Hold Out method
kneigh.predict(new_X) ## => new_X : vector of feature to predict
```

# 6 Model Tuning  
  
## 6.1 Grid Search  

> Grid Search with KNN Classifier
```python
from sklearn.model_selection import GridSearchCV
knn_class = KNeighborsClassifier()
grid = {'n_neighbors':[5,10,20,30]}  ## => Hyperparameter Grid
grid_search = GridSearchCV(knn_class, grid, scoring = 'accuracy', cv = 5, n_jobs=-1) # paralellize computation 
grid_search.fit(X_train,y_train);
print(f"1. Best Score :  {grid_search.best_score_}")  ## => Best Score regarding scoring defined above
print(f"2. Params (K):{grid_search.best_params_}")
print(f"3. Best Estimator : {grid_search.best_estimator_}")
```  
  
  
## 6.2 Random Search  
  
> Random Search with KNN Classifier
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn_class = KNeighborsClassifier()
random_param = {'n_neighbors':[1, 5, 10, 15, 20], 'p':[1,2,3]} ## => Hyperparameter Grid
random_search = RandomizedSearchCV(knn_class, random_param, scoring = 'recall', n_iter=5, cv = 5, n_jobs=-1) # paralellize computation 
random_search.fit(X_train,y_train);
print(f"1. Best Score :  {random_search.best_score_}")  ## => Best score (related to scoring metric)
print(f"2. Params (K):{random_search.best_params_}")  ## => Best params
print(f"3. Best Estimator : {random_search.best_estimator_}")  
```  

# 7. Data Sourcing
## 7.1 API

```python
import requests

BASE_URI = "<insert base URI/URL>" ## => Look into API Documentation to find BASE_URI
path = "<insert path>" ## => Look into API Documentation to find path of requested data
params = {
    'query': <insert query>,
    'format': 'json'  ## => if needed to specify the format for the API call
}
response = requests.get(BASE_URI+path, params).json()
```

## 7.2 Web Scraping

```python
import requests
from bs4 import BeautifulSoup

BASE_URI = "<insert base URI/URL>" ## => Place the URL you want to Scrape
path = "<insert path>" ## => behind URL if needed to scrape several pages for example
response = requests.get(BASE_URI+path, headers={"<Put some headers>"})  ## => headers could be needed to specify language content in return
soup = BeautifulSoup(response.content, "html.parser")

## iterate inside soup to find the corresponding tag in which is stored the desired content.
## Example below for books
for book in soup.find_all("article", class_="product_pod"):
    title = book.find("h3").find("a").string
    price = float(book.find(class_="price_color").string.strip('£'))
    for rate, value in ratings.items():
        if book.find(class_="star-rating "+rate) != None:
            rating = value
    title_list.append(title)
    price_list.append(price)
    rating_list.append(rating)

```