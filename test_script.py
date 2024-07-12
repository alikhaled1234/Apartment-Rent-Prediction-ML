import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import math
from sklearn.metrics import r2_score
from sklearn import metrics

pd.set_option('display.max_columns', None)

def man_dist(x1, y1, x2, y2):
  return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    
def get_nearest_city_with_ll(cn):
    min_dist = 100000
    cityy = ""
    for index, row in cn.iterrows():
        for i, city in enumerate(cities_l):
            if i == 0:
                min_dist = man_dist(row['latitude'], row['longitude'], cities_l[city][0], cities_l[city][1])
                cityy = city
            else:
                
                if min_dist > man_dist(row['latitude'], row['longitude'], cities_l[city][0], cities_l[city][1]):
                    min_dist = man_dist(row['latitude'], row['longitude'], cities_l[city][0], cities_l[city][1])
                    cityy = city
                    
                    
    df_test.iloc[index, df_test.columns.get_loc('cityname')] = cityy
    
    
def get_nearest_city_with_s(cn):
    for index, row in cn.iterrows():
        
        if math.isnan(row['state']):
            
            df_test.iloc[index, df_test.columns.get_loc('cityname')] = cities_m
            df_test.iloc[index, df_test.columns.get_loc('state')] = states_m
            continue
        for city in cities_s:
            if cities_s[city] == row['state']:
                df_test.iloc[index, df_test.columns.get_loc('cityname')] = city
        if math.isnan(df_test.iloc[index, df_test.columns.get_loc('cityname')]):
            df_test.iloc[index, df_test.columns.get_loc('cityname')] = cities_m
    
    
def get_nearest_state_with_c(cn):
    
    for index, row in cn.iterrows():
        for state in states_s:
            if states_s[state] == row['cityname']:
                df_test.iloc[index, df_test.columns.get_loc('state')] = state
        
        if type(df_test.iloc[index, df_test.columns.get_loc('state')]) == float:
            
            df_test.iloc[index, df_test.columns.get_loc('state')] = states_m






df_test = pd.read_csv(r'D:\ml-datasets project ms1\apartment test data\ApartmentRentPrediction_test.csv')

with open('bathrooms_mode.pkl', 'rb') as f:
    bath = pickle.load(f)
with open('bedrooms_mode.pkl', 'rb') as f:
    room = pickle.load(f)
with open('feet.pkl', 'rb') as f:
    feet = pickle.load(f)
with open('cities_l.pkl', 'rb') as f:
    cities_l = pickle.load(f)
with open('cities_s.pkl', 'rb') as f:
    cities_s = pickle.load(f)
with open('states_l.pkl', 'rb') as f:
    states_l = pickle.load(f)
with open('states_s.pkl', 'rb') as f:
    states_s = pickle.load(f)
with open('cities_m.pkl', 'rb') as f:
    cities_m = pickle.load(f)
with open('states_m.pkl', 'rb') as f:
    states_m = pickle.load(f)
with open('unique_amenities.pkl', 'rb') as f:
    unique_amenities = pickle.load(f) 
with open('bounds_dict.pkl', 'rb') as f:
    bounds_dict = pickle.load(f) 
with open('state_order.pkl', 'rb') as f:
    state_order = pickle.load(f) 
with open('min_max_long.pkl', 'rb') as f:
    min_max_long = pickle.load(f) 
with open('min_max_sqr_feet.pkl', 'rb') as f:
    min_max_sqr_feet = pickle.load(f) 
with open('RF.pkl', 'rb') as f:
    RF = pickle.load(f) 
    
with open('XGB.pkl', 'rb') as f:
    XGB = pickle.load(f) 
    
with open('SVM.pkl', 'rb') as f:
    SVM = pickle.load(f) 
   


df_test['bathrooms'] = df_test['bathrooms'].fillna(bath)

if df_test['bedrooms'].isna().sum() + df_test['square_feet'].isna().sum() > 0:

    studios = df_test[df_test['title'].str.contains(r'studio', case = False)]
    df_test['type'] = np.zeros((df_test.shape[0], 1))
    villas = df_test[df_test['title'].str.contains(r'\bvilla\b', case = False)]
    df_test.iloc[studios.index, -1] = 1
    df_test.iloc[villas.index, -1] = 2
    
    for i in range(0, 3):
      df_test.loc[df_test['type'] == i, 'bedrooms'] = df_test.loc[df_test['type'] == i, 'bedrooms'].fillna(room[i])
      df_test.loc[df_test['type'] == i, 'square_feet'] = df_test.loc[df_test['type'] == i, 'square_feet'].fillna(feet[i])
      
    
    df_test = df_test.drop('type', axis = 1)
     
    
    
    
if df_test['cityname'].isna().any():
    city_name_nulls = df_test[df_test['cityname'].isna()]
    if city_name_nulls['longitude'].isna().sum() + city_name_nulls['latitude'].isna().sum() > 0:
        #print(df_test['cityname'].isna().sum())
        city_name_null_no_l = city_name_nulls[(city_name_nulls['longitude'].isna()) | (city_name_nulls['latitude'].isna())]
        city_name_null_l = city_name_nulls[~((city_name_nulls['longitude'].isna()) | (city_name_nulls['latitude'].isna()))]
        get_nearest_city_with_ll(city_name_null_l)
        #print(df_test['cityname'].isna().sum())
        get_nearest_city_with_s(city_name_null_no_l)
        #print(df_test['cityname'].isna().sum())
    else:
        get_nearest_city_with_ll(city_name_nulls)
    df_test['cityname'] = df_test['cityname'].fillna(cities_m)
        
    
if df_test['state'].isna().any():
    get_nearest_state_with_c(df_test[df_test['state'].isna()])
    
if df_test['latitude'].isna().any():
    lat_null = df_test[df_test['latitude'].isna()]
    for index, row in lat_null.iterrows():
        if cities_l.get(row['cityname']) is not None:
            df_test.iloc[index, df_test.columns.get_loc('latitude')] = cities_l[row['cityname']][0]
    
if df_test['longitude'].isna().any():
    long_null = df_test[df_test['longitude'].isna()]
    for index, row in long_null.iterrows():
        if cities_l.get(row['cityname']) is not None:
            df_test.iloc[index, df_test.columns.get_loc('longitude')] = cities_l[row['cityname']][1]

    
df_test.loc[(~df_test['body'].str.contains(r'\bamenit|\butilit', case = False)) & (df_test['amenities'].isna()), 'amenities'] = '0'
df_test.loc[(df_test['body'].str.contains(r'\bamenit|\butilit', case = False)) & (df_test['amenities'].isna()), 'amenities'] = '1'


df_test['amenities'] = df_test['amenities'].str.split(',')


df_test['amenities'] = df_test['amenities'].fillna('1')


for amenity in unique_amenities:
    df_test[amenity] = df_test['amenities'].apply(lambda x: 1 if amenity in x else 0)

# Dropping the original amenities column
df_test = df_test.drop(columns=['amenities'])


cols = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']



for col in cols:
    
    df_test.loc[(df_test[col] > bounds_dict[col]['upper']), col] = bounds_dict[col]['upper']
    df_test.loc[(df_test[col] < bounds_dict[col]['lower']), col] = bounds_dict[col]['lower']




ordinal_encoder = OrdinalEncoder(categories = state_order, handle_unknown='use_encoded_value', unknown_value=-1)
df_test['state'] = ordinal_encoder.fit_transform(df_test[['state']])


target = 'price'

y = df_test[target]

df_test = df_test[['bathrooms', 'bedrooms', 'square_feet', 'longitude', 'state', 'Parking', 'Dishwasher',
         'Garbage Disposal', 'Internet Access', '0', 'Playground', 'Cable or Satellite', 'Patio/Deck', 'Pool']]

y = np.log1p(y)

df_test['longitude'] = ((df_test['longitude'] - min_max_long[1]) / (min_max_long[0] - min_max_long[1]) * (6)) - 3
df_test['square_feet'] = ((df_test['square_feet'] - min_max_sqr_feet[1]) / (min_max_sqr_feet[0] - min_max_sqr_feet[1]) * (6)) - 3

print(df_test.describe())

df_test.head()

predicitons = RF.predict(df_test)


r2 = r2_score(y, predicitons)
mse = metrics.mean_squared_error(y, predicitons)

print(r2, " ", mse)


preds = XGB.predict(df_test)


r2_model6 = r2_score(y, preds)
mse_model6 = metrics.mean_squared_error(y, preds)

print(r2_model6, " ", mse_model6)


    
    
    