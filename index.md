## Welcome to My GitHub Pages

I am a complex systems physicist with a particular interest in interdisciplinary projects. I am interested in artificial intelligence, evo-devo and education. You can see my artistic work on Instagram blog ([@fabienfrfr](https://www.instagram.com/fabienfrfr/)) and a prototype of neural game in Itch.io [@fabienfrfr](https://fabienfrfr.itch.io/).

See me Heroku Python Web app [here](https://fabienfrfr.herokuapp.com)

# WORK IN PROGRESS ...

# Titre

Enoncé du problème et objectif général


```python
import os
import pandas as pd, numpy as np
import pylab as plt, seaborn as sns
import geopandas as gpd, networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# exploration
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

# learning
import sklearn as sk
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing as skpp
from sklearn import model_selection as skms
from sklearn import metrics as skm
from sklearn import feature_selection as skfs

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# save
import joblib
%store -r plot_var

# param
pd.set_option('display.max_row', 200)
pd.set_option('display.max_column', 200)
```

---

## 1. Data Collection

Link INSEE, France Data, Kaggle Dataset, WebScrapling, Yahoo Finance...

### 1-a. Data knowledge

Wikipedia

### 1-b. Data overview

- Spreadsheet vizualization
- Variable list (scale, dependency, type)
- Objectif mesurable

### 1-c. Data importation


```python
dataf = pd.read_csv(os.getcwd() + '/file.csv')
dataf = pd.read_excel('path.excel', sheet_name=['name'])
geodf = gpd.read_file("path.shp")
```


```python
## basic statistic
data.describes() # all
print(dataf['columns'].sum())
dataf.groupby(['columns1','columns2']).agg(['mean','std']) # pivot table of specifics feature
```

### 1-d. Data pre-analysis


```python
## new feature
dataf['new_columns'] = dataf['columns A'] / dataf['columns B']
dataf.insert(1, 'new_columns', dataf[['A','B']].idxmax(1)) # feature 2 labels variables
## specifics visualization
geodf = geodf.merge(dataf[['ON', 'new']], how='inner', on='ON')
geo_df[geo_df.ON.str[:2] != "97"].plot(ax = ax, categorical=True, column = "Results", legend=True)
#### simplify dataset (bonus : scipy.interpolate)
data = data.drop(columns=columns_list)
""" you can create columns by group "Missing indicator" if empty info (ex : confidential data), it's becaume data information """
# interpolate data & verification
data = (data1['var_list'] + data2['var_list']) / 2 # good order (simplify data)
sns.catplot(data = pd.concat([data1['specVar'],data['specVar'],data2['specVar']], axis=1), kind="violin")
```

### 1-e. Data compilation


```python
for d in dataf_list :
    data = pd.merge(data, d, how='inner', on='ON') # automatics also with time series
# head or tail
data.head()
```

## 2. Data Exploration

Find interesting variables for analysis


```python
df = data.copy()
```

### 2-a. Data definition

Variable target :


```python
print(df.shape) # row and columns
print(df.dtypes.value_counts()) # variable type
print((df.isna().sum()/df.shape[0]).sort_values(ascending=True)) # NaN proportion
# for time series : see resample per date, rolling/ewn, FFT analysis (+possible filter)
```

### 2-b. Data relationship


```python
corr = df.select_dtypes('float').corr()
grid = sns.clustermap(corr) # correlation map linked by average method and euclidean metric
```


```python
# complete relation ship (max columns = 20, sample dataset by 1000)
sns.pairplot(df.sample(1000), hue="label") # histogram and all variable/variable scatter
#sns.jointplot() # 2d histogram
```

### 2-c. Data reduction visualization


```python
# standardization
x = df.loc[:, features].values
x = preprocessing.StandardScaler().fit_transform(x)
# PCA compression of continious variables (if unlinear : Manifold : IsoMap)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
pc_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])
# adding category
pc_df['cat'] = df['label'].astype("category")
# 3D plot (contourf possible)
#ax = plt.axes(projection='3d')
#ax.scatter(pc_df['A'], pc_df['B'], pc_df['C'], c= pc_df['cat'])
sns.scatterplot(data = pc_df, hue = "cat")
```

### 2-c. Data contingency


```python
contigency = pd.crosstab(df['label_A'], df['label_B'])
sns.heatmap(contigency, annot=True)
```

### 2-d. Data Test


```python
# test of independence
c, p, dof, expected = chi2_contingency(contigency) 
# H0 hypothesis (student)
stat, p = ttest_ind(df_rowsA['columns'].dropna(), df_rowsB['columns'].dropna())
# show variable
print(f'{'' :-<50} {if p < 0.02 : 'H0 Rejetée'}')
```

## 3. Data preprocessing

Prepare data for analysis and modelization (sous ensemble)


```python
df_ = df.copy()
df_ = df[df.isna().sum()/df.shape[0] < 0.85] # only columns with no relation, completly empty or no variance (constance)
```

### 3-a. Dataset training-test construction


```python
trainset, testset = train_test_split(df_, test_size=0.2, random_state=0)
# proportion of class repartition (try to distribute with the same proportion)
trainset['label'].value_counts()
```

### 3-b. Data Encoding


```python
# convert string to values (possible use "Transformer" LabelEncoder or onehot+sparseMatrix if normalization class)
code = {'A':0,'B':1,'C':0}
df.loc[:,'Label'] = df[col].map(code) # lambda function also possible, otherwithe replace(list,list) or .astype("category").cat.codes
```

--- 
*Return here after basic test modelization (Trial and error method)*

### 3-c. Data Transform



```python
def preprocessing(df):
    ### feature engineering (underfitting)
    df['new'] = df[] # ex : polynomial feature "optimize/extraction", binarizer, kernel log/log. Oversampling :
    # unsupervised_info = Kmean clustering class (+Elbow method for good seeder), PCA axis, Isolation Forest (Anomaly)
    ### feature imputation (overfitting) - SimpleImputer(strategy='')
    df = df.dropna() # drop-fill-NA, KNNImputer, image morphology (artefact delete). feature selection :
    # selector = VarianceThreshold() # constante columns, other : selectKbest (Khi-2), SelectFromModel (if parameter model, KNN not included)
    # input / target
    X = df.drop('label', axis=1)
    y = df['label']
    return X,y
```

## 4. Data modelization

**Steps :**
- 1. Dataset convention :  matrice for input : x_feature^{exemple}
- 2. Model and parameter : y = ax + b. (a,b)
- 3. Cost Function (RMS criteria) --> evaluation
- 4. Minimization algorithm (iteratif : SGD, direct : Eq. Normale "par inversion de matrice")

*Other method equiv for linear :* covariance matrix calculation, but ungeneralized.-


```python
X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)
# if not dataframe (numpy array), add shape if (N,) to be (1,N) : reshape or [None]
# warning of numpy broadcoasting if numpy : (1,3) + (3,1) = (3,3)
```

### 4-a. Basic model testing


```python
def evalutation(model) :
    # fit
    model.fit(X_train, y_train) # step 3,4 directly included
    # first evaluation
    #model.score(X,y) #R2 automatics metric if regression
    # predict
    ypred = model.predict(X_test)
    # evaluation
    print(confusion_matrix(y_test, ypred)) #(sum=support) nb_success A, nb_error A ## nb_succes B, nb_error B
    print(classification_report(y_test, ypred))
    # learning and validation curve (with cross validation : Kfold per default)
    N, train_score, val_score = learning_curve(model, X_train, y_train,cv=4, scoring='f1',train_sizes=np.linspace(0.1, 1, 10))
    # learning curve F1 = precision/recall (see overfit, underfit)
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
```


```python
# import
test_model = DecisionTreeClassifier(random_state=0) # first
#test_model = RandomForestClassifier(random_state=0) # regularized model (basic ensemble)
#test_model = make_pipeline(PolynomialFeatures(2), SelectKBest(f_classif, k=10), RandomForestClassifier(random_state=0)) # here contain data transform included in model
# evaluation
evalutation(test_model)
```

**Visualization of feature importance (for selection)**


```python
pd.DataFrame(test_model.feature_importances_, index=X_train.columns).plot.bar(figsize=(12, 8))
```

---
*When the precision, recall and f1-score is > 50% (convergence of Law of large numbers), the dataset it's good*

### 4-b. Pipeline ensemble learning

Pipeline = Transformer+Estimator chain

Ensemble learning = Law of large numbers + Competence + Diversity


```python
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))
#preprocessor = make_columns_selector((categorical_pipeline, ['cat1','...','catn']),(numerical_pipeline, ['v1','...','vn']))
```


```python
# classification model (basic = logistic regression)
RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
# regression model (basic = linear regression)
"""
LinearRegression
etc.
"""
```


```python
dict_of_models = {'RandomForest': RandomForest, # bagging : all in overfitting grouping result
                  'AdaBoost' : AdaBoost, # boosting : all in underfitting grouping result
                  'SVM': SVM,
                  'KNN': KNN
                 }
""" Note : Stacking it's for predictor of predictor, VotingClassifier method for bagging/boosting without diversity """
```


```python
for name, model in dict_of_models.items():
    print(name)
    evaluation(model)
```

**Model choisi :** 

### 4-c. Hyperparameter search


```python
# for chosen model 
""""
# 1st for grid
hyper_params = {'svc__gamma':[1e-3, 1e-4],
                'svc__C':[1, 10, 100, 1000, 3000]}
"""
hyper_params = {'svc__gamma':[1e-3, 1e-4, 0.0005],
                'svc__C':[1, 10, 100, 1000, 3000], 
               'pipeline__polynomialfeatures__degree':[2, 3],
               'pipeline__selectkbest__k': range(45, 60)}
# last it's not adapted for grid (too many parameter)
```


```python
# grid = GridSearchCV(SVM, hyper_params, scoring='recall', cv=4) # first for some parameter
grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4, n_iter=40) # cv = crossvalidation (cutting in 4 validation set of train set here)
grid.fit(X_train, y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
```


```python
evaluation(grid.best_estimator_)
```

### 4-d. Threshold search


```python
precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()
```

### 4-e. Final model


```python
def model_final(model, X, threshold=0):
    return model.decision_function(X) > threshold
```


```python
y_pred = model_final(grid.best_estimator_, X_test, threshold=-1)
# score for classification model
f1_score(y_test, y_pred)
recall_score(y_test, y_pred)
# score for regression model
"""
mean_absolute_error(y_test, y_pred) # linear problem
mean_squared_error(y_test, y_pred) # exponential problem
median_absolute_error(y_test, y_pred) # if some outlier = robust (warning : high risk)
"""
```

## 5. Conclusion

Chosen model, link between other report, good variable

### 5-a. Synthetic visualization

*Relative to dataset :* geographic overview, pairplot, reduction_observation, feature_selection, evaluation_curve


```python
#fig = plt.figure(figsize=(12,12))
#gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
#gridkw = dict(height_ratios=[5, 1])
#fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw=gridkw)

#data[5][data[5].CODGEO.str[:2] != "97"].plot(ax=ax, categorical=True, column = "Results", legend=True)
#sns.clustermap(data=corr, xticklabels=True) #, ax=gs[0,0])
#pd.DataFrame(feature_weight, index=X_train.columns).plot.bar(figsize=(12, 8))
"""
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
"""
```

### 5-b. Discussion

Possible amelioration (if more time)

---

### Supplementary


```python
# store basic variable after closing session (for seeing curve faster)
plot_var = {'df': 0 }
%store plot_var
```

---



You can use the [editor on GitHub](https://github.com/fabienfrfr/fabienfrfr.github.io/edit/main/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/fabienfrfr/fabienfrfr.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
