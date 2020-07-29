import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')



# read the data
data = pd.read_csv('MarketingData.csv')

# shape and data types of the data
print(data.shape)
print(data.dtypes)

# % of missing.
percent_missing = data.isnull().sum() * 100 / len(data)

# Delete first column
data = data.drop(data.columns[0], axis=1)

# variance of each column
variance_data = data.var()

#delete the columns that have 0 variance 
data1 = data.loc[:,data.apply(pd.Series.nunique) != 1]



# Delete column which contains missing value more than % 50
cols = data1.columns[data1.isnull().mean()>0.5]
df = data1.drop(cols, axis=1)


# Delete rows which contains less than 1000 non NaN values
df = df.dropna(axis =0 , thresh=1000)

#remove duplicate observations
df= df.drop_duplicates()


#list of colomn names
colomn_names = df.columns.values.tolist()

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)


# percent of missing categorical values
percent_missing_cat = df_non_numeric.isnull().sum() * 100 / len(df_non_numeric)

# percent of missing numerical values
percent_missing_num = df_numeric.isnull().sum() * 100 / len(df_numeric)



# Create correlation matrix
corr_matrix = df_numeric.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
df_numeric.drop(to_drop, axis=1, inplace=True)




#correlation with target variable 
corr_target = df_numeric.corrwith(df_numeric["target"]).abs()

#deleting columns that have less than 0.01 correlation with target variable
threshold= 0.01

corr = pd.Series(df_numeric.corrwith(df_numeric["target"]).abs()) > threshold 


df_numeric = df_numeric[df_numeric.columns[corr]]



#unique values of categorcal variables
unique_non_numeric = df_non_numeric.nunique()

#unique values of numeric variables
unique_numeric = df_numeric.nunique()



#deleting redundant columns
threshold2= 2000

u_value = pd.Series(unique_non_numeric) < threshold2 


df_non_numeric = df_non_numeric[df_non_numeric.columns[u_value]]


unique_non_numeric.sort_values(ascending=False )


df_non_numeric= df_non_numeric.drop(['sciera_ilec_availability', 'sciera_ilec_provider', 'serloc_dma_name','serloc_previous_region_name',
                     'serloc_previous_division_name','serloc_headend_name','serloc_dwelling_type_reporting_desc'], axis=1)




# impute the missing values for each numeric column.
# df_numeric = df.select_dtypes(include=[np.number])

numeric_cols = df_numeric.columns.values

for col in numeric_cols:
    missing = df_numeric[col].isnull()
    num_missing = np.sum(missing)
    
    if num_missing > 0:  # only do the imputation for the columns that have missing values.
        med = df_numeric[col].median()
        df_numeric[col] = df_numeric[col].fillna(med)
        
        
# impute the missing values for each non-numeric column.
# df_non_numeric = df.select_dtypes(exclude=[np.number])

non_numeric_cols = df_non_numeric.columns.values

for col in non_numeric_cols:
    missing = df_non_numeric[col].isnull()
    num_missing = np.sum(missing)
    
    if num_missing > 0:  # only do the imputation for the columns that have missing values.        
        top = df_non_numeric[col].describe()['top'] # impute with the most frequent value.
        df_non_numeric[col] = df_non_numeric[col].fillna(top)
        
 

#list of numeric colomn names
col_names_num = df_numeric.columns.values.tolist()
#list of non_numeric colomn names
col_names_non_num = df_non_numeric.columns.values.tolist()



# Feature Scaling

       
a = df_numeric.iloc[:, 0:666].values

b =pd.DataFrame(data= df_numeric.iloc[:, 666].values)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a = sc.fit_transform(a)
a = pd.DataFrame(data = a)


df_numeric = pd.concat([a, b], axis=1)

#naming colomn names
df_numeric.columns = col_names_num




#Dummies encoding

# Get dummies
df_non_numeric = pd.get_dummies(df_non_numeric, prefix_sep='_', drop_first=True)
# X head
df_non_numeric.head()



#########################################################################

#OneHot encoding

# import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# instantiate OneHotEncoder
ohe = OneHotEncoder(categorical_features = df_non_numeric, sparse=False ) 
# categorical_features = boolean mask for categorical columns
# sparse = False output an array not sparse matrix

# apply OneHotEncoder on categorical feature columns
X_ohe = ohe.fit_transform(df_non_numeric) # It returns an numpy array




#DictVectorizer encoding

# turn X into dict
X_dict = df_non_numeric.to_dict(orient='records') # turn each row as key-value pairs
# show X_dict
X_dict


# DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# instantiate a Dictvectorizer object for X
dv_X = DictVectorizer(sparse=False) 
# sparse = False makes the output is not a sparse matrix

# apply dv_X on X_dict
X_encoded = dv_X.fit_transform(X_dict)
# show X_encoded
X_encoded

# vocabulary
vocab = dv_X.vocabulary_
# show vocab
vocab





#label encoding

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
df_non_numeric[col_names_non_num] = df_non_numeric[col_names_non_num].apply(lambda col: le.fit_transform(col))
df_non_numeric[col_names_non_num].head(10)

##################################################################3





df_numeric.reset_index(drop=True, inplace=True)
df_non_numeric.reset_index(drop=True, inplace=True)

#merging the numerical and categorical dateset
result = pd.concat([df_non_numeric, df_numeric], axis=1)






#Calculating cumulative_explained_variance

from sklearn import decomposition 
from sklearn.decomposition import PCA


pca = decomposition.PCA()

pca.n_components = 2707

pca_data = pca.fit_transform(result)

percentage_var_explained = pca.explained_variance_/ np.sum(pca.explained_variance_) 

cum_var_explained = np.cumsum(percentage_var_explained)

#plot the PCA spectrum

plt.figure(1,figsize=(6,4))

plt.clf()
plt.plot(cum_var_explained, linewidth =2 )
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('cumulative_explained_variance')
plt.show





#PCA

from sklearn.decomposition import PCA


model = PCA(n_components=500).fit(result)
X_pc = model.transform(result)

# number of components
n_pcs= model.components_.shape[0]

# get the index of the most important feature on EACH component i.e. largest absolute value
# using LIST COMPREHENSION HERE
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

colomn_names2 = result.columns.values.tolist()

# get the names
most_important_names = [colomn_names2[most_important[i]] for i in range(n_pcs)]

# using LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df1 = pd.DataFrame(sorted(dic.items()))





################################################################
#Applying PCA

X = result.iloc[:, 0:2706].values
y = result.iloc[:, 2706].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling(it has been applied already)
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


from sklearn.decomposition import PCA
pca = PCA(n_components = 500)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


#plt.plot(range(400), pca.explained_variance_ratio_)
plt.plot(range(500), np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Explained Variance")




##################################################################

# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



##########################################################################################

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5] }

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)



import scipy
scipy.test()



param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

##########################################################################################

# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = classifier.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))





