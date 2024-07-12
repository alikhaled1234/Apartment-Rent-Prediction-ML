import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('D:\ml-datasets project ms1\ApartmentRentPrediction.csv')

"""# Data Exploration"""

df.head()

df.info()

df.describe()

df.columns

"""making all text lower case"""

df['title'] = df['title'].str.lower()
df['body'] = df['body'].str.lower()

df['title'] = df['title'].str.replace('one', '1')
df['body'] = df['body'].str.lower()

"""changing all letters writtin numbers to digits"""

numbers = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6'}


def number_converter(data, col):
    new_sentences = []
    for sentence in data[col]:
        words = sentence.split()
        new_words = []
        for word in words:
            if word in numbers:
                word = numbers[word]
            new_words.append(word)
        new_sentences.append(' '.join(new_words))
    return new_sentences


df['title'] = number_converter(df, 'title')
df['body'] = number_converter(df, 'body')

"""price_display can't be object"""

df['price_display'] = df['price_display'].str.replace('$', '')
df['price_display'] = df['price_display'].str.replace(',', '')

df.loc[df['price_display'].str.contains(r'[a-z]'), 'price_display'] = \
    df.loc[df['price_display'].str.contains(r'[a-z]'), 'price_display'].str.split(' ').str[0]

df['price_display'] = df['price_display'].astype('int64')

"""category analysis"""

df['category'].value_counts()

sns.countplot(data=df, x='category')

"""should be dropped

amenities analysis
"""

df['amenities'].value_counts()

"""
bathroom analysis
"""

df['bathrooms'].value_counts()

sns.countplot(data=df, x='bathrooms', color='y')
plt.xlabel('bathrooms')
plt.show()

"""

*       1 and 2 bathrooms are the most common number of bathrooms

"""

sns.barplot(data=df, x='bathrooms', y='price')

"""

*   as number of bathroom increases the price inc.
*   the floating number bathroom count may indicate small bathroom without all the utilities
*   all bathrooms more then 4 have small count and most likely outliers or wrong data

"""

sns.barplot(data=df, x='bathrooms', y='square_feet')

"""apartments with 8 or more bathrooms are massive

**bedrooms analysis**
"""

df['bedrooms'].value_counts()

sns.countplot(data=df, x='bedrooms')

"""0 bedrooms is strange"""

sns.barplot(data=df, x='bedrooms', y='square_feet')

"""apartments with more than 5 bedrooms are exeptionally big"""

sns.barplot(data=df, x='bedrooms', y='price_display')

"""*   0 bedroom apartments have high price with respect to other apartments(maybe they are special aparments)
*   apartments with (6, 7, 8) bedrooms are very expensive (maybe luxurious apartments)

*   there is only one apartment with 9 rooms and with small amount of money

**Rooms Analysis**
"""

df['rooms'] = df['bathrooms'] + df['bedrooms']

# sns.barplot(data = df, x = 'bathroom', y = 'price_display', color='blue', label='Barplot 1')
# sns.barplot(data = df, x = 'bedroom', y = 'price_display', color='orange', label='Barplot 2', bottom='bathroom')


# df['rooms'].value_counts()

sns.countplot(data=df, x='rooms')

df.groupby('rooms')['price_display'].count().index

sns.barplot(data=df, x='rooms', y='price_display')
plt.xticks(rotation=75)

sns.barplot(data=df, x='rooms', y='square_feet', order=df['rooms'].value_counts().index)
plt.xticks(rotation=25)

"""**currency**"""

df['currency'].value_counts()

sns.countplot(data=df, x='currency')

"""this columns has only one value so it has no value to us

**fee**
"""

df['fee'].value_counts()

sns.countplot(data=df, x='fee')

"""this columns has only one value so it has no value to us

**has_photo**
"""

df['has_photo'].value_counts()

sns.countplot(data=df, x='has_photo')

sns.barplot(data=df, x='has_photo', y='price_display')

"""


*  Having as photo doesn't necessarily affect the price


"""

sns.barplot(data=df, x='has_photo', y='square_feet')

"""**pets_allowed analysis**"""

df['pets_allowed'].value_counts()

sns.countplot(data=df, x='pets_allowed')

"""
*   Cats are usually more accepted in apartments than dogs
"""

sns.barplot(data=df, x='pets_allowed', y='price_display')

"""**Price display analysis**"""

sns.histplot(data=df, x='price_display', kde=True)

"""*   big right tail (skewed)
*   there are outliers

**price type analysis**
"""

df['price_type'].value_counts()

sns.countplot(data=df, x='price_type')

"""*   There isn't much variability
*   should be dropped

**square feet analysis**
"""

sns.histplot(data=df, x='square_feet', kde=True)

"""

*   big right tail (skewed)
*   there are outliers

"""

sns.scatterplot(data=df, x='square_feet', y='price_display')

"""there is correlation between the 2 features

**address analysis**
"""

df['address'].value_counts()

"""as expected most of them have unique address except for few of them

**cityname analysis**
"""

df['cityname'].value_counts()

sns.countplot(data=df[df['cityname'].isin(df['cityname'].value_counts().head(15).index)], x='cityname')
plt.xticks(rotation=75)

sns.barplot(data=df[df['cityname'].isin(['Austin', 'Dallas', 'Houston', 'San Antonio', 'Los Angeles', 'Chicago'])],
            x='cityname', y='price_display',
            order=df[df['cityname'].isin(['Austin', 'Dallas', 'Houston', 'San Antonio', 'Los Angeles', 'Chicago'])][
                'cityname'].value_counts().index)

"""**state analysis**"""

df['state'].value_counts()

plt.figure(figsize=(10, 10))
sns.countplot(data=df, x='state', order=df['state'].value_counts().index)
plt.xticks(rotation=75)

plt.figure(figsize=(10, 10))
sns.barplot(data=df, x='state', y='price_display', order=df['state'].value_counts().index)
plt.xticks(rotation=75)

sns.catplot(data=df, x='state', y='price_display', order=df['state'].value_counts().index, height=10)
plt.xticks(rotation=75)

"""**latitude & longitude analysis**"""

sizes = []
sizes = (df['price_display'])

sns.scatterplot(data=df, x='longitude', y='latitude', size=sizes, hue=sizes)

"""**source analysis**"""

df['source'].value_counts()

sns.countplot(data=df, x='source', order=df['source'].value_counts().index)
plt.xticks(rotation=75)

sns.barplot(data=df, x='source', y='price_display', order=df['source'].value_counts().index)
plt.xticks(rotation=75)

sns.catplot(data=df, x='source', y='price_display', order=df['source'].value_counts().index)
plt.xticks(rotation=75)

sns.barplot(data=df, x='source', y='square_feet', order=df['source'].value_counts().index)
plt.xticks(rotation=75)

"""since the rest of the sources have small count we can join them"""

df['source'] = np.where((df['source'] != 'RentLingo') & (df['source'] != 'RentDigs.com'), 'other', df['source'])

sns.barplot(data=df, x='source', y='price_display', order=df['source'].value_counts().index)
plt.xticks(rotation=75)

"""**time analysis**"""

sns.histplot(data=df, x='time', kde=True)

sns.scatterplot(data=df, x='time', y='price_display')

df.columns

"""# **Preprocessing**"""

df.isna().sum()

df['id'].duplicated().any()

"""split the data into numerical and categorical"""

num = ['id', 'price', 'price_display', 'square_feet',
       'latitude', 'longitude', 'time']

cat = ['bathrooms', 'bedrooms', 'category', 'title', 'body', 'amenities', 'currency', 'fee',
       'has_photo', 'pets_allowed', 'price_type', 'address', 'cityname',
       'state', 'source']

"""----------------------------------------------------------------------------------------------"""

df['category'].value_counts()

df['category'] = df['category'].str.split('/').str[2]

"""low variability in this column, should be dropped

getting all apartments that are studios
"""

studios = df[df['title'].str.contains(r'studio', case=False)]
villas = df[df['title'].str.contains(r'\bvilla\b', case=False)]

df['type'] = np.zeros((df.shape[0], 1))

df.iloc[studios.index, -1] = 1
df.iloc[villas.index, -1] = 2

"""## **Nulls handling**"""

"""bathroom

nulls
"""

df['bathrooms'].value_counts()

bathNull = df[df['bathrooms'].isna()]
bathNull[bathNull['body'].str.contains(r'\bBa\b|bathroom', case=False)]['body']

bathNull['body'].str.extractall(r'(\D{8}bathroom\D{20}|ba\b)')

"""it has one bathroom"""

df.iloc[4779, 5] = 1
df.iloc[7741, 5] = 1

"""fill the rest by the mode"""
bathrooms_mode = df['bathrooms'].fillna(df['bathrooms'].mode()[0])
df['bathrooms'] = bathrooms_mode

with open('bathrooms_mode.pkl', 'wb') as f:
    pickle.dump(bathrooms_mode, f)

"""**bedroom**

Nulls
"""

studios['bedrooms'].isna().sum()

"""studios usually dont contain bedrooms"""

villas['bedrooms'].isna().sum()

"""no villa has nulls

fill the bedroom nulls with the mode of the same type of apartment
"""

df.loc[(df['type'] == 1) & (df['bedrooms'].isna())] = df.loc[(df['type'] == 1) & (df['bedrooms'].isna())].fillna(0)

bedrooms_mode = df.loc[df['bedrooms'].isna()].fillna(df.loc[df['type'] == 2, 'bedrooms'].mode()[0])

df.loc[df['bedrooms'].isna()] = bedrooms_mode


roomm = []
feet = []
for i in range(0, 3):
    roomm.append(df.loc[df['type'] == i, 'bedrooms'].mode()[0])
    feet.append(df.loc[df['type'] == i, 'square_feet'].mode()[0])


with open('bedrooms_mode.pkl', 'wb') as file:
    pickle.dump(roomm, file)


with open('feet.pkl', 'wb') as f:
    pickle.dump(feet, f)

"""amenities

nulls
"""

df['amenities'].value_counts()

df['amenities'].isna().sum()

amen_nulls = df[df['amenities'].isna()]

amen_nulls[amen_nulls['title'].str.contains(r'\bamenit\b|\butilit\b', case=False)].shape[0]

"""no titles contain extra amenities"""

amen_nulls['body'].str.extractall(r'(\bamenit|\butilit)').shape[0]

amen_nulls['body'].str.extractall(r'(\bamenit.{3,7}are.{1,50}|\butilit.{3,10}are.{1, 50})').shape[0]

amen_nulls['body'].str.extractall(r'(\bamenit.{3,10}:.{1,50}|\butilit.{3,10}:)').shape[0]

df.loc[(~df['body'].str.contains(r'\bamenit|\butilit', case=False)) & (df['amenities'].isna()), 'amenities'] = '0'
df.loc[(df['body'].str.contains(r'\bamenit|\butilit', case=False)) & (df['amenities'].isna()), 'amenities'] = '1'

df['amenities'] = df['amenities'].str.split(',')

df['amenities'].isna().sum()

df['amenities'] = df['amenities'].fillna('1')

unique_amenities = set()
for amenities in df['amenities']:
    if (amenities != '1') & (amenities != '0'):
        unique_amenities.update(amenities)

for amenity in unique_amenities:
    df[amenity] = df['amenities'].apply(lambda x: 1 if amenity in x else 0)

# Dropping the original amenities column
df.drop(columns=['amenities'], inplace=True)

"""**pets_allowed**

handling nulls
"""

df['pets_allowed'].isna().sum()

df['pets_allowed'].value_counts()

pets_nulls = df[df['pets_allowed'].isna()]

pets_nulls['body'].str.extractall(r'(pets are not)|(no pet)')

"""there are apartments that dont allow pets"""

df.iloc[pets_nulls['body'].str.extractall(r'(pets are not)|(no pet)').index.get_level_values(0), 10] = '0'

df['pets_allowed'] = np.where(df['pets_allowed'] == 1.0, '1', df['pets_allowed'])
df['pets_allowed'] = np.where(df['pets_allowed'] == 0, '0', df['pets_allowed'])

"""the rest null values can be set as cats and dogs"""

df['pets_allowed'] = df['pets_allowed'].fillna('Cats,Dogs')

df.isna().sum()

"""**Address**

Nulls
"""

# address_nulls = df[df['address'].isna()]

# address_nulls

# address_nulls['body'].str.extractall(r'(this unit is located at.*)')

df['address'] = df['address'].fillna("not provided")

"""**cityname**

Nulls
"""

cities = df[~df['cityname'].isna()]['cityname'].unique()

cities_nulls = df[(df['cityname'].isna())]


def get_name(data, col):
    for city_null in data[col]:
        for city in cities:
            if city in city_null:
                print('hna')


get_name(df, 'title')

get_name(df, 'body')

"""city name isn't stated in title or body, we can estimate it from other rows near enough using latitude and longitude"""


def get_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


dict = {}
city_index = df.columns.get_loc('cityname')
for index, c in cities_nulls.iterrows():
    for oc_index, oc in df[~df['cityname'].isna()].iterrows():
        if dict.get(index) is None:
            dict[index] = get_distance(c['longitude'], c['latitude'], oc['longitude'], oc['latitude'])

            df.iloc[index, city_index] = oc['cityname']

        else:
            if dict[index] > get_distance(c['longitude'], c['latitude'], oc['longitude'], oc['latitude']):
                dict[index] = get_distance(c['longitude'], c['latitude'], oc['longitude'], oc['latitude'])
                df.iloc[index, city_index] = oc['cityname']

"""**State**

Nulls
"""

state_nulls = df[df['state'].isna()]

"""we can replace those with the city's state"""

state_nulls['cityname'].unique()

state_nulls_replacement = {}


for col in state_nulls['cityname'].unique():
    state_mode = df.loc[(df['state'].isna() == False) & (df['cityname'] == col), 'state'].mode()
    if not state_mode.empty:
            state_nulls_replacement[col] = state_mode[0]
            # Fill NaN state values with the mode state value
            df.loc[(df['state'].isna()) & (df['cityname'] == col), 'state'] = state_mode[0]

states_l = {}
states_s = {}

df['state'].isna().sum()

"""**latitude**

Nulls
"""

latitude_nulls = df[df['latitude'].isna()]

latitude_nulls

"""they are all from the same city and same state, so we can use the mean of all cities in the same area"""

cary_latitude = df.loc[df['cityname'] == 'Cary', 'latitude'].mean()
cary_longitude = df.loc[df['cityname'] == 'Cary', 'longitude'].mean()

df.loc[df['latitude'].isna(), 'latitude'] = cary_latitude
df.loc[df['longitude'].isna(), 'longitude'] = cary_longitude
'''
location_data = {
    'latitude': df['latitude'].mean(),
    'cary_longitude': df['longitude'].mean()
}

with open('cary_location.pkl', 'wb') as f:
    pickle.dump(location_data, f)
'''
df.isna().sum()


    
cities_l = {}
cities_s = {}
states_l = {}
states_s = {}
cities_m = df['cityname'].mode()[0]
states_m = df['state'].mode()[0]

for city in df['cityname'].unique():
  cities_l[city] = (df.loc[df['cityname'] == city, 'latitude'].mean(), df.loc[df['cityname'] == city, 'longitude'].mean())
  cities_s[city] = df.loc[df['cityname'] == city, 'state'].mode()[0]

for state in df['state'].unique():
  states_l[state] = (df.loc[df['state'] == state, 'latitude'].mean(), df.loc[df['state'] == state, 'longitude'].mean())
  states_s[state] = df.loc[df['state'] == state, 'cityname'].mode()[0]

    

with open('cities_l.pkl', 'wb') as f:
    pickle.dump(cities_l, f)
    
with open('cities_s.pkl', 'wb') as f:
    pickle.dump(cities_s, f)
    
with open('states_l.pkl', 'wb') as f:
    pickle.dump(states_l, f)
    
with open('states_s.pkl', 'wb') as f:
    pickle.dump(states_s, f)

with open('cities_m.pkl', 'wb') as f:
    pickle.dump(cities_m, f)
    
with open('states_m.pkl', 'wb') as f:
    pickle.dump(states_m, f)

with open('unique_amenities.pkl', 'wb') as f:
    pickle.dump(unique_amenities, f)


"""All nulls are handled

min_max_long = (df['longitude'].max(), df['longitude'].min())


## **Outliers Detection and Visulization**
"""

outlier = ['bathrooms', 'bedrooms', 'price_display', 'square_feet', 'latitude', 'longitude']

for feat in outlier:
    fig, axis = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(data=df, x=feat, ax=axis[0])
    sns.histplot(data=df, x=feat, ax=axis[2])
    sns.scatterplot(data=df, x=feat, y='price_display', ax=axis[1])
    plt.xlabel(feat)
    plt.show

"""**Inspecting the outliers**"""

"""Price display"""

df[df['price_display'] > 15000]

"""the price is exeptionally high maybe there is an additional wrong zero that needs to be removed"""

df.loc[df['price_display'] > 15000, 'price_display'] = df.loc[df['price_display'] > 15000, 'price_display'] / 10


# finding outliers
def interquartile(data, col):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    return data[(data[col] > data[col].quantile(0.75) + (IQR * 1.5)) | (
            data[col] < data[col].quantile(0.25) - (IQR * 1.5))].shape[0], \
        data[col].quantile(0.75) + (IQR * 1.5), data[col].quantile(0.25) - (IQR * 1.5)


for col in outlier:
    print(col, " : ", interquartile(df, col))

# df = df2.copy()

"""**impute outliers by the interquartile range**"""


# imputing outliers
def get_bounds(data, col):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    return data[col].quantile(0.75) + (IQR * 1.5), data[col].quantile(0.25) - (IQR * 1.5)

#7ot price_display
cols = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']

bounds_dict = {}

for col in cols:
    upper, lower = get_bounds(df, col)
    df.loc[(df[col] > upper), col] = upper
    df.loc[(df[col] < lower), col] = lower
    bounds_dict[col] = {'upper': upper, 'lower': lower}

with open('bounds_dict.pkl', 'wb') as f:
    pickle.dump(bounds_dict, f)

for col in outlier:
    print(col, " : ", interquartile(df, col))

"""No outliers except in price column which is acceptable

# **Feature Engineering**
"""

df['total_rooms'] = df['bathrooms'] + df['bedrooms']

"""# **Feature Encoding**"""

categorical_cols = ['category', 'currency', 'fee', 'price_type', 'cityname']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = LabelEncoder.fit_transform(le, df[col])

cats = ['has_photo', 'pets_allowed', 'state', 'source']

df['has_photo'].unique()

print(df.groupby('state')['price_display'].mean().sort_values(ascending=False).index)

df.isna().sum()

order = [['No', 'Thumbnail', 'Yes'],  # Example order for feature 1
         ['0', '1', 'Dogs', 'Cats', 'Cats,Dogs'],
         ['HI', 'CA', 'MA', 'DC', 'RI', 'NJ', 'WA', 'NY', 'MT', 'MD', 'OR', 'FL',
          'IL', 'VA', 'VT', 'NH', 'CO', 'MN', 'NV', 'AL', 'TN', 'ID', 'SC', 'MI',
          'CT', 'UT', 'WI', 'DE', 'PA', 'GA', 'NC', 'AZ', 'TX', 'IN', 'OH', 'WV',
          'MS', 'LA', 'OK', 'AR', 'AK', 'KS', 'MO', 'NM', 'KY', 'IA', 'NE', 'ND',
          'SD', 'WY'],
         ['RentDigs.com', 'RentLingo', 'other']]

ordinal_encoder = OrdinalEncoder(categories=order)

df[cats] = ordinal_encoder.fit_transform(df[cats])



state_order = [['HI', 'CA', 'MA', 'DC', 'RI', 'NJ', 'WA', 'NY', 'MT', 'MD', 'OR', 'FL',
          'IL', 'VA', 'VT', 'NH', 'CO', 'MN', 'NV', 'AL', 'TN', 'ID', 'SC', 'MI',
          'CT', 'UT', 'WI', 'DE', 'PA', 'GA', 'NC', 'AZ', 'TX', 'IN', 'OH', 'WV',
          'MS', 'LA', 'OK', 'AR', 'AK', 'KS', 'MO', 'NM', 'KY', 'IA', 'NE', 'ND',
          'SD', 'WY']]

with open('state_order.pkl', 'wb') as f:
    pickle.dump(state_order, f)

df.head()

"""# **Feature Selection**"""
"""we can drop the following:


1.   id : doesn't have value for us
2.   category, price_type, fee, currency: they have low to no variability(all have the same value, no variance)
3.   body, title: already extracted useful info from them
4.   Address: each apartment has different address
5.   cityname: already its information is stated in the state column

"""

df = df.drop(columns=['id', 'category', 'price_type', 'fee', 'currency', 'body', 'title', 'price', 'address', 'cityname'])

df.head()

df = df[['bathrooms', 'bedrooms', 'has_photo', 'pets_allowed',
         'square_feet', 'state', 'latitude', 'longitude', 'source', 'time',
         'type', 'Gym', 'Alarm', 'Parking', 'Dishwasher', 'Garbage Disposal',
         'Doorman', 'Clubhouse', 'Fireplace', 'Hot Tub', 'Storage', 'View', 'AC',
         'Tennis', 'Washer Dryer', 'Internet Access', '0', 'Elevator', 'TV',
         'Basketball', 'Luxury', 'Playground', '1', 'Golf', 'Cable or Satellite',
         'Patio/Deck', 'Wood Floors', 'Pool', 'Refrigerator', 'Gated',
         'total_rooms', 'price_display']]

df[['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'time', 'total_rooms', 'price_display']].corr()

sns.heatmap(df[['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'time', 'total_rooms',
                'price_display']].corr(), annot=True)

"""drop latitude, time and total_rooms"""

df = df.drop(columns=['time', 'total_rooms'])

"""**Anova test**"""


c = ['has_photo', 'pets_allowed',
     'state', 'source', 'type', 'Gym', 'Alarm', 'Parking',
     'Dishwasher', 'Garbage Disposal', 'Doorman', 'Clubhouse', 'Fireplace',
     'Hot Tub', 'Storage', 'View', 'AC', 'Tennis', 'Washer Dryer',
     'Internet Access', '0', 'Elevator', 'TV', 'Basketball', 'Luxury',
     'Playground', '1', 'Golf', 'Cable or Satellite', 'Patio/Deck',
     'Wood Floors', 'Pool', 'Refrigerator', 'Gated']
X = df.loc[:, c]

selector = SelectKBest(score_func=f_regression, k=10)  # Select top 10 features
selector.fit(X, df['price_display'])

selected_features_indices = selector.get_support(indices=True)

print(selector.pvalues_, X.columns)

np.sort(selector.pvalues_)

"""The final features"""

df = df[['bathrooms', 'bedrooms', 'square_feet', 'longitude', 'state', 'Parking', 'Dishwasher', 'Garbage Disposal',
         'Internet Access',
         '0', 'Playground', 'Cable or Satellite', 'Patio/Deck', 'Pool', 'price_display']]

"""# **Feature Scaling**"""

sns.histplot(data=df, x='longitude')

sns.histplot(data=df, x='square_feet')

sns.histplot(data=df, x='price_display')


def scale_features(df, col, m):
    if m == 0:
        min_val = df[col].min()
        max_val = df[col].max()
        return ((df[col] - min_val) / (max_val - min_val) * (6)) - 3
    elif m == 1:
        mean_val = df[col].mean()
        std_val = df[col].std()
        return (df[col] - mean_val) / std_val
    else:
        return np.log1p(df[col])

min_max_long = (df['longitude'].max(), df['longitude'].min())
min_max_sqr_feet = (df['square_feet'].max(), df['square_feet'].min())

with open('min_max_long.pkl', 'wb') as f:
    pickle.dump(min_max_long, f)


with open('min_max_sqr_feet.pkl', 'wb') as f:
    pickle.dump(min_max_sqr_feet, f)
"""Normalization"""

df['price_display'] = scale_features(df, 'price_display', 2)

sns.histplot(data=df, x='price_display')

# df = df.drop(df[(df['price_display'] > 9) | (df['price_display'] < 6)].index, axis = 0)

df['square_feet'] = scale_features(df, 'square_feet', 0)

sns.histplot(data=df, x='square_feet')

# df = df.drop(df[(df['square_feet'] > 8.5) | (df['square_feet'] < 5.5)].index, axis = 0)

"""standardization"""

df['longitude'] = scale_features(df, 'longitude', 0)

sns.histplot(data=df, x='longitude')

# df = df.drop(df[df['longitude'] < -1].index, axis = 0)

"""# Model"""
pd.set_option('display.max_columns', None)


df.columns

X = df.copy().drop(columns=['price_display'], axis=1).to_numpy()  # Features
Y = df['price_display'].copy().to_numpy()
Y = Y.reshape((X.shape[0], 1))

# X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['price_display'], axis=1), df['price_display'], test_size=0.20, shuffle=True, random_state=10)


poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X_train)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_poly, y_train)

X_poly_test = poly.transform(X_test)

y_pred = model.predict(X_poly_test)

r2_model1 = r2_score(y_test, y_pred)
mse_model1 = metrics.mean_squared_error(y_test, y_pred)
print("LinearRegression R2 score: ", r2_model1)
print("LinearRegression MSE: ", mse_model1)

with open('PR.pkl', 'wb') as f:
    pickle.dump(model, f)

model2 = RandomForestRegressor(n_estimators=100, random_state=10)

# Step 5: Fit Model
model2.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model2.predict(X_test)

r2_model2 = r2_score(y_test, y_pred)
mse_model2 = metrics.mean_squared_error(y_test, y_pred)
print("RandomForestRegressor R2 score: ", r2_model2)
print("RandomForestRegressor MSE: ", mse_model2)

with open('RF.pkl', 'wb') as f:
    pickle.dump(model2, f)

model3 = XGBRegressor(n_estimators=75, learning_rate=0.4, random_state=10)

# Step 5: Fit Model
model3.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model3.predict(X_test)

r2_model3 = r2_score(y_test, y_pred)
mse_model3 = metrics.mean_squared_error(y_test, y_pred)
print("XGBRegressor R2 score: ", r2_model3)
print("XGBRegressor MSE: ", mse_model3)

with open('XGB.pkl', 'wb') as f:
    pickle.dump(model3, f)

model4 = SVR(kernel='rbf')

# Step 6: Fit Model
model4.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model4.predict(X_test)

r2_model4 = r2_score(y_test, y_pred)
mse_model4 = metrics.mean_squared_error(y_test, y_pred)
print("SVR R2 score: ", r2_model4)
print("SVR MSE: ", mse_model4)

with open('SVM.pkl', 'wb') as f:
    pickle.dump(model4, f)

model5 = DecisionTreeRegressor(random_state=10)

# Step 5: Fit Model
model5.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model5.predict(X_test)

r2_model5 = r2_score(y_test, y_pred)
mse_model5 = metrics.mean_squared_error(y_test, y_pred)
print("DecisionTreeRegressor R2 score: ", r2_model5)
print("DecisionTreeRegressor MSE: ", mse_model5)

with open('DT.pkl', 'wb') as f:
    pickle.dump(model5, f)

model6 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.4, random_state=10)

# Step 5: Fit Model
model6.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model6.predict(X_test)

r2_model6 = r2_score(y_test, y_pred)
mse_model6 = metrics.mean_squared_error(y_test, y_pred)
print("GradientBoostingRegressor R2 score: ", r2_model6)
print("GradientBoostingRegressor MSE: ", mse_model6)

with open('GB.pkl', 'wb') as f:
    pickle.dump(model6, f)
    
    

p = model2.predict(df.drop('price_display', axis = 1))
model2.get_params()
r2_model6 = r2_score(df['price_display'], p)
r2_model6
mse_model6 = metrics.mean_squared_error(f['price_display'], p)
