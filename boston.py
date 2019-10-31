import os	#to bring operating system functionalities so to join and get file paths 
import tarfile	#to extract tar files accessed from web
from six.moves import urllib	#to get python3 and python2 features combined
import pandas as pd	#for data analysis
%matplotlib inline
import matplotlib.pyplot as plt	#for data visualization
import numpy as np	#data manipulation and maths
import hashlib	#for hash md5 particularly
from pandas.plotting import scatter_matrix	#pandas.tools.plotting is no longer supported instead use pandas.plotting

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"	#download link
HOUSING_PATH = os.path.join("datasets","housing")	#downlad path in local computer
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"	#final url for accessing

#function to get file from the url and get it as .tar file then extract it and store it as .csv in the defined path
def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
#finction to get the csv file and read it using pandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

#the fetch function needs to be called first so as to create the defined path and bring in the .tar file in the system
#then the load function is called which returns a dataframe
fetch_housing_data()
housing = load_housing_data()

#then check for the dataframe's head
#on checking the dataframe we get 10 attributes and 20640 different districts
#longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,and ocean_proximity .
housing.head()

#gives the information about each attribute
#noticed that the total_bedrooms attribute has only 20433 non-null values meaning 207 districts are missing this feature.
#ocean_proximity is string type attribute.
housing.info()

#gives the different values in ocean_proximity attribute and there count.
housing["ocean_proximity"].value_counts()

#gives the summary of the numerical attributes.
#null values are not counted.
#we get the standard deviation and median
housing.describe()

#show histogram of data
# Notice a few things in these histograms:
# 1. First, the median income attribute does not look like it is expressed in US dollars (USD). After
# checking with the team that collected the data, you are told that the data has been scaled and capped
# at 15 (actually 15.0001) for higher median incomes, and at 0.5 (actually 0.4999) for lower median
# incomes. Working with preprocessed attributes is common in Machine Learning, and it is not
# necessarily a problem, but you should try to understand how the data was computed.
# 2. The housing median age and the median house value were also capped. The latter may be a serious
# problem since it is your target attribute (your labels). Your Machine Learning algorithms may learn
# that prices never go beyond that limit. You need to check with your client team (the team that will use
# your system’s output) to see if this is a problem or not. If they tell you that they need precise
# predictions even beyond $500,000, then you have mainly two options:
# a. Collect proper labels for the districts whose labels were capped.
# b. Remove those districts from the training set (and also from the test set, since your system should
# not be evaluated poorly if it predicts values beyond $500,000).3. These attributes have very different scales. We will discuss this later in this chapter when we
# explore feature scaling.
# 4. Finally, many histograms are tail heavy: they extend much farther to the right of the median than to
# the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. We
# will try transforming these attributes later on to have more bell-shaped distributions.
housing.hist(bins=50, figsize=(20,15))
plt.show()	

#create a test set and never look back at it.Generally 20% of data is selected.
#function to digest current index into hashvalue
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1]<256 * test_ratio

#function to split the dataframe into test and train data
def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_Set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

#split the dataframe into train and test sets
housing_with_id = housing.reset_index()
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")

#create a new income_cat col
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)

#we can also use the train_test_split function provided in sklearn package
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing["income_cat"].value_counts()/len(housing)

for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis=1, inplace=True)

#to keep our testset untouched
housing = strat_train_set.copy()

#since it is a geographical data a scatter plot will be good
housing.plot(kind="scatter", x="longitude", y="latitude")

#to better visualize the density of datapoints
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

#to look at the housing prices and there density
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100,label="population",figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()

#for correlation among attributes
corr_matrix = housing.corr()

#to know the correlation between median_house_value vs other attributes
corr_matrix["median_house_value"].sort_values(ascending=False)

#create a list of attributes and draw a scatter plot of these attributes
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

#plot the scatter plot showing corelation among the income and house value attribute
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)

#create new attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#check corelation with the new attributes
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#to seperate features and lable
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#drop total_bedroom attribute from housing and add median 
housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms",axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)

#use sklearn.impute import SimpleImporter instead of sklearn.preprocessing import Imputer
#imputer here specifies that you want to replace each atrribute's missing values with the median of that attribute
from sklearn.impute import SimpleImputer
imputer  = SimpleImputer(strategy="median")

#since median can only be computed on numerical values we need to drop ocean_proximity
housing_num = housing.drop("ocean_proximity",axis=1)

#fit the imputer instance to the training data usinf the fit() method
imputer.fit(housing_num)


