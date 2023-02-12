# tuning
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler #標準化
from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
seed = 55688
sc = StandardScaler()
path = r'/Users/mllab/Desktop/DATA/整理資料/procrastination_MAX_3_dummy.csv'
# path = r'/Users/mllab/Desktop/DATA/整理資料/procrastination_impute_cart_std.csv'
# path = r'/Users/mllab/Desktop/DATA/整理資料/procrastination_day_.csv'
# path = r'/Users/mllab/Desktop/DATA/整理資料/procrastination_impute_rf_std.csv'

df =  pd.read_csv(path,
                  header = None,
                  encoding = 'utf-8')
df.head()

# x = df.iloc[1:500,[3,4,5,6,7,8,13,14,15,16,17,18,19]] # 有新增finish標籤
# x = df.iloc[1:500,[3,4,5,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29]] # 有新增finish標籤 #dummy
# x = df.iloc[1:500,[4,16,18,23]] # 特徵選取 3分
x = df.iloc[1:500,[3,4,5,13,14,16,17,18,23,29]] # 特徵選取 3分+2分
y = df.iloc[1:500,[20]]                  #標籤 22=70分,23=60分,24=50分
# le = LabelEncoder()
# x[6]=le.fit_transform(x[6])
# x[7]=le.fit_transform(x[7])
# x[8]=le.fit_transform(x[8])
# x[19]=le.fit_transform(x[19])
from sklearn.model_selection import train_test_split
#data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=seed, stratify=y, shuffle=True)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
cv = StratifiedKFold(n_splits=10, random_state= 851206, shuffle=True)

#def scorer

from sklearn.metrics import make_scorer, recall_score, accuracy_score, f1_score, precision_score, mean_squared_error
acc = make_scorer(accuracy_score)
recall = make_scorer(recall_score,pos_label="1")
f1 = make_scorer(f1_score,pos_label="1")
precision = make_scorer(precision_score,pos_label="1")
mean_squared_error = make_scorer(mean_squared_error)
scorer = recall


#SVM
# pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=55688))
svc = SVC(random_state=seed)
svc2 = SVC(random_state=seed,probability=True)
param_range_svc_C = [1,10,50,100,1000,5000,10000,50000,100000]
param_range_svc_gamma = [0.000001,0.00001,0.0001,0.001,0.01]
tol_range = [0.00001,0.00005,0.0001,0.0005,0.001,0.01,0.1]
param_range_svc_kernel = ["rbf"]
param_grid_svc = dict(C = param_range_svc_C, gamma = param_range_svc_gamma, 
                      kernel = param_range_svc_kernel,
                      tol = tol_range)
gs_svc = RandomizedSearchCV(estimator=svc, param_distributions=param_grid_svc,scoring=scorer, cv=cv,n_jobs=-1,n_iter=500)
# gs_svc_2 = RandomizedSearchCV(estimator=svc2, param_distributions=param_grid_svc,scoring=scorer,cv=cv,n_jobs=-1,n_iter=200)
# gs_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc,scoring='accuracy',cv=5,n_jobs=-1)

gs_svc = gs_svc.fit(x_train,y_train.ravel())
# gs_svc_2 = gs_svc_2.fit(x_train,y_train.ravel())


#KNN
knn = KNeighborsClassifier()
param_range_knn_n = [3,4,5,6,7,8,9]
param_range_knn_weight = ["distance",'uniform']
param_range_knn_algorithm = ["auto"]
param_range_knn_metric = ['minkowski','euclidean','manhattan','nan_euclidean','haversine','cosine']
param_range_knn_leafsize = [30]
param_grid_knn = dict(n_neighbors = param_range_knn_n,weights=param_range_knn_weight, metric=param_range_knn_metric, 
                      leaf_size = param_range_knn_leafsize)
gs_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_grid_knn,scoring=scorer,cv=cv,n_jobs=-1,n_iter=500)
# gs_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn,scoring='accuracy',cv=5,n_jobs=-1)

gs_knn = gs_knn.fit(x_train,y_train.ravel())


#LogisticRegression
from sklearn.linear_model import LogisticRegression
LogisticR = LogisticRegression(max_iter=1000,random_state=seed)
penalty_options = ['l2']
C_range = [1,10,100,1000,5000,10000,50000,100000,500000]
tol_range = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01]
solver_options = ['lbfgs']
param_grid_lr = dict(penalty = penalty_options, C=C_range,tol = tol_range, solver=solver_options)
gs_lr = RandomizedSearchCV(estimator=LogisticR, param_distributions=param_grid_lr,scoring=scorer,cv=cv,n_jobs=-1,n_iter=500)
# gs_lr = GridSearchCV(estimator=LogisticR, param_grid=param_grid_lr,scoring='accuracy',cv=5,n_jobs=-1)

gs_lr = gs_lr.fit(x_train,y_train.ravel())


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
n_components_option = [1]
solver_options = ['svd','lsqr','eigen']
tol_range = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
param_grid_lda = dict(solver = solver_options,tol = tol_range, n_components = n_components_option)
gs_lda = RandomizedSearchCV(estimator=lda, param_distributions=param_grid_lda,scoring=scorer,cv=cv,n_jobs=-1,n_iter=500)
# gs_lda = GridSearchCV(estimator=lda, param_grid=param_grid_lda,scoring='accuracy',cv=5,n_jobs=-1)
s_lda = gs_lda.fit(x_train,y_train.ravel())


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = seed)
n_estimators_range = [100, 150, 200, 250, 300, 350, 400, 450, 500,550,600,650]
criterion_option = ['gini','entropy']
max_depth_range = [2,3,4,5,6,7,8,9,10]
max_features_range = [1,2,3,4,5,6]
min_samples_split_range = [2,3,4,5,6,7,8,9,10]
param_grid_rf = dict(n_estimators = n_estimators_range, criterion = criterion_option, 
                      max_depth = max_depth_range, max_features = max_features_range, min_samples_split = min_samples_split_range)
gs_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf,scoring=scorer,cv=cv,n_jobs=-1,n_iter=500)
# gs_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf,scoring='accuracy',cv=5,n_jobs=-1)
gs_rf = gs_rf.fit(x_train,y_train.ravel())

print(gs_svc.best_score_)
print(gs_svc.best_params_)
print(gs_knn.best_score_)
print(gs_knn.best_params_)
print(gs_lr.best_score_)
print(gs_lr.best_params_)
print(gs_lda.best_score_)
print(gs_lda.best_params_)
print(gs_rf.best_score_)
print(gs_rf.best_params_)



