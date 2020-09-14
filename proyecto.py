import pandas
import numpy
import sklearn
import os
import re
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, vstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt  #use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)



def parseStr(value):
    value = value.lower()
    value = re.sub('[^a-zA-Z]+', '', value)
    return value.strip() 

def getValueForCurrency(row, currencies):
    return currencies[row['cur_name']][0] *  row['mp_price']

def concatenateCountryMKT(row):
    country = parseStr(row['adm0_name'])
    locality = parseStr(row['adm1_name'])
    mkt = parseStr(row['mkt_name'])
    return country + '/' + locality  + '/' + mkt 

def vectorizeMatrix(columns_to_vectorize, df, enc, result):
    # enc = DictVectorizer()
    if(not result):
        X_train_categ = enc.fit_transform(df[columns_to_vectorize].to_dict('records'))
    else:
        X_train_categ = enc.transform(df[columns_to_vectorize].to_dict('records'))

    years = df['mp_year'].values
    months = df['mp_month'].values
    X = hstack([X_train_categ])
    return numpy.vstack((X.A.T,years,months)).T



os.chdir(os.path.dirname(__file__))

currencies = pandas.read_csv('curr.csv', encoding='latin-1')
currencies = currencies.set_index('currency').T.to_dict('list')
train = pandas.read_csv('wfp_market_food_prices.csv', index_col=0, encoding='latin-1')
train = train.sort_index()
train = train.sample(frac = 1) 
print(f'Shape of train csv : {train.shape}')

to_drop =['adm1_id', 'mkt_id','cm_id', 'cur_id','pt_id','um_id', 'mp_commoditysource', 'pt_name']
train.drop(columns=to_drop, axis=1, inplace=True)
train = train.dropna()
train = train.reset_index(drop = True) 

train['mkt'] = train.apply (lambda row: concatenateCountryMKT(row), axis=1)
train['cm_name'] = train.apply (lambda row: parseStr(row['cm_name']), axis=1)
train['um_name'] = train.apply (lambda row: row['um_name'].lower().strip(), axis=1)
train['price_usd'] = train.apply (lambda row: getValueForCurrency(row, currencies), axis=1)

scalerYears = MinMaxScaler()
scalerMonths = MinMaxScaler()
train['mp_year'] = scalerYears.fit_transform(train['mp_year'].values.reshape(-1,1))
train['mp_month'] = scalerMonths.fit_transform(train['mp_month'].values.reshape(-1,1))


to_drop =['adm0_name', 'adm1_name','mkt_name', 'cur_name', 'mp_price']
train.drop(columns=to_drop, axis=1, inplace=True)
train = train.dropna()
train = train.reset_index(drop = True) 

print(f'Data insight')
plt.scatter(train['mp_year'].values, train['price_usd'].values)
plt.xlabel('Year')
plt.ylabel('Price in USD')
plt.show()


plt.scatter(train['mp_month'].values, train['price_usd'].values)
plt.xlabel('Month')
plt.ylabel('Price in USD')
plt.show()


print(f'Pre proccessed train csv shape : {train.shape}')
print(train)


columns_to_vectorize= ['mkt', 'cm_name', 'um_name']

enc = DictVectorizer()
C= vectorizeMatrix(columns_to_vectorize, train, enc, False)

y = train['price_usd']

X_train, X_test, y_train, y_test = train_test_split(C, y, test_size=0.3, random_state=123)
print(f'X train shape : {X_train.shape}  X test shape : {X_test.shape} T Y train shape: {y_train.shape}  Y test shape: {y_test.shape} ')

# Classifier: 
clf = Ridge(alpha=.23, random_state=123)
clf.fit(X_train, y_train) 

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"train_score {train_score}")
print(f"test_score {test_score}")

print(f'Querying the following dataframe...')

query = pandas.read_csv('options_to_test.csv',encoding='latin-1')
query['mp_year'] = scalerYears.transform(query['mp_year'].values.reshape(-1,1))
query['mp_month'] = scalerMonths.transform(query['mp_month'].values.reshape(-1,1))
print(query.head())
columns_to_vectorize_in_query= ['mkt', 'cm_name', 'um_name']

sparse_matrix = vectorizeMatrix(columns_to_vectorize_in_query,query, enc, True)
result = clf.predict(sparse_matrix)
query['predicted_price'] = result
print(query)
