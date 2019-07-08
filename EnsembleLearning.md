#### Ensemble Learning 

[TOC]



Weak Models >> Voting >> Final Output

##### Voting

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-8vS4rH2fShSXGWgBpIpAyk0Qy2T0pCxb7D-WccPrkDdEGPvbrg)

```python
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier( estimators=
[ ('log_clf', LogisticRegression()),
 ('svm_clf', SVC(probability=True),#default is False
 ('tree_clf', DescisionTressClassifer(random_stata = 666))   
],
  voting = 'hard/soft/...'                        )

voting_clf.fit(xtrain, ytrain)
voting_clf.score(xtest, ytest)
```



voting question>> the number of primary models is not large enough, 

therefore, we consider about bagging methods to do estimations for the primary models. Thus, the voting process will have good performance.



methods for ensemble learning combination >> by mean, voting, learning

bagging, decision tree/random forest/ extremely random forest

boosting, adaboost/gradient boosting

stacking,



##### Bagging

sampling methods:

bagging, with replacement >> in stat, called bootstrap

pasting, without replacement



```python
## Bagging Method
from sklearn.tree import DescisionTreeClassifier
from sklearn.ensemble import BaggingClassifer
bagging_clf = BaggingClassifier(
DescisionTreeClassifier(),
n_estimators=500, #Number of trees
max_samples = 100, # each sample size
bootstrap = True)  #True>> replacement, False>> non-replacement
bagging_clf.fit(xtrain, ytrain)
bagging_clf.score(xtest, ytest)
```



replacement re-sample may contribute to about 37% samples out of bagging

therefore, use oob to reduce possibly bias 

```python
#oob out-of-bag
bagging_clf = BaggingClassifier(
DescisionTreeClassifier(),
n_estimators=500, #Number of trees/models
max_samples = 100, # each sample size
bootstrap = True, #True>> replacement, False>> non-replacement
oob_score = True)  # avoid the miss bagged samples
bagging_clf.fit(x, y) # here, x, y are whole (train+test) dataset
bagging_clf.oob_score_ # output the fitted score
```



##### n_jobs

n_jobs to accelerate the fitting process, 可并行的

```python
# n_jobs
bagging_clf = BaggingClassifier(
DescisionTreeClassifier(),
n_estimators=500, 
max_samples = 100, 
bootstrap = True, 
oob_score = True,  
n_jobs = -1)
bagging_clf.fit(x, y)
```

Random Subspaces #target in features

Random Patches # features & samples



##### bootstrap_features

```python
#bootstrap_subspaces, the random featrue subspaces
random_subspaces_clf = BaggingClassifier(
DescisionTreeClassifier(),
n_estimators=500, #Number of models
max_samples = ..., # each sample size should be equals to size of rows, which means without sampling the samplespace
bootstrap = True, #True>> replacement, False>> non-replacement
oob = True,
n_jobs = -1, # num of features
max_features =..., # number of features
bootstrap_features =True)
```

```python
# bootstrap patches， the random sample subspaces and feature subspaces
# usually in CV
random_patches_clf = BaggingClassifier(
DescisionTreeClassifier(),
n_estimators=500, #Number of trees
max_samples = 100, # sample size for each subspace
bootstrap = True, #True>> replacement, False>> non-replacement
oob = True,
n_jobs = -1, # num of features
max_features =1, # 
bootstrap_features =True)
```

##### Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, random_state = 666, oob_score= True, n_jobs = -1)
rf_clf.fit(x, y)
rf_clf.oob_score_

rf_clf2 = RandomForestClassifier(n_estimators = 500, max_leaf_node = 16, random_state=666, oob_score = True, n_jobs = -1) # adding the max leaf of node to avoid too many classify nodes
rf_clf2.fit(x, y)
rf_clf2.oob_score_
```



##### Extra-Trees (Extremely Random Forest)

extremely random forest, faster and avoid overfitting, however, increasing bias.

```python
from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(n_estimators = ..., bootstrap = True, oob_score = True, random_state=...)
et_clf.fit(x, y)
et_clf.oob_score_
```



##### Ada Boost

Redefine the weights for different sample points based on last sub model, each new sub model intents to boost the error from the latest model, finally, by voting each model to get a final output. In terms of algorithm, the redefinition for weight can be transformed into finding the extremely values.

![](https://i1.wp.com/adataanalyst.com/wp-content/uploads/2016/08/AdaBoost.jpg?fit=239%2C200)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth =.., n_estimators=..., )
```



##### Gradient Boosting

Gradient Boosting >> train model M1 and get error e1, then for e1 , training M2, and get e2, >>then train M3 based on e2 and get e3..., the final output will be m1+m2+...

![](https://pbs.twimg.com/media/Co9mfXbWYAAooqy.jpg)

![](http://uc-r.github.io/public/images/analytics/gbm/boosted_stumps.gif)



```python
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(max_depet= ...,n_estimators= ....)
```

```python
## can also be use on regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
```



##### Stacking

![](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor_files/stackingregression_overview.png)

Train dataset1 to train model1 and output the predictions then these predictions can become another dataset for the following model training dataset



##### Ensemble Learning for Regression 

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreeRegressor
......
```

