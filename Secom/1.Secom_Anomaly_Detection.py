#0.라이브러리
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score

#1.데이터 로드
df = pd.read_csv('Secom/data/uci-secom.csv')

#2.EDA
print(df.info())
print(df)
print(df.select_dtypes(exclude='float'))
print(df['Pass/Fail'].value_counts())
df_view = pd.DataFrame(df['Pass/Fail'].value_counts())
sns.barplot(data=df_view,x='Pass/Fail',y='count',hue='Pass/Fail', legend=False)
plt.show()
sns.scatterplot(data=df,x='Time',y='Pass/Fail') # Pass/Fail 분포 확인
plt.xticks([])
plt.show()

# Pass/Fail과 독립변수에 대한 상관관계 분석
print(df.drop(columns='Time').corr()['Pass/Fail'].abs().sort_values(ascending=False).head(11))
df_view = pd.DataFrame(data=df.drop(columns='Time').corr()['Pass/Fail'].abs().sort_values(ascending=False).head(11))[1:].reset_index()
df_view.rename(columns={'index': 'div', 'Pass/Fail': 'value'}, inplace=True)
sns.barplot(data=df_view,x='div',y='value', legend=False)
for index, row in df_view.iterrows():
    plt.text(row.name, round(row.value,3), round(row.value,3), color='black', ha="center")
plt.xlabel('')
plt.ylabel('')
plt.show()
corr_matrix = df[['59','103','510','348','158','431','Pass/Fail']].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, linewidths=0.5)
plt.show()

sns.pairplot(df[['59','103','510','348','158','431','Pass/Fail']])
plt.show()

#3.데이터 정제
# 불필요한 컬럼 제거
df = df.drop(columns='Time')

# 고유값이 1개인 컬럼에 대해 NaN를 -1로 대체
# print((df.loc[:,df.nunique()==1].sum()<0).sort_values(ascending=False))
columns_nunique_1 = df.loc[:,df.nunique()==1].columns
for col in columns_nunique_1:
        df.loc[df[col].isna(),col]=-1

# 결측치 많은 컬럼 제거
# print(df.isna().sum().sort_values(ascending=False).head(60))
df = df.loc[:,(df.isna().sum()>len(df)*0.4)==False]

# 다중 대체법을 활용한 결측치 처리
model = RandomForestRegressor(random_state=999)
imputer = IterativeImputer(model, max_iter=20, random_state=999)

df_sub = df.loc[:,~df.columns.isin(np.append(columns_nunique_1,'Pass/Fail'))]

for col in df_sub.columns:
        imputer_col_list_droped = df_sub.corr()[col].sort_values(ascending=False).drop(col).head(3).index.tolist()
        imputer_col_list = [col]+imputer_col_list_droped
        print(imputer_col_list)

        imputer_fit = imputer.fit(df_sub[imputer_col_list])
        imputed_data = imputer_fit.transform(df_sub[imputer_col_list])
        
        df[col] = imputed_data[:,0]

# df.to_csv('Secom/data/df_IterativeImputer20.csv', index=True) # 결측치 처리까지 진행된 데이터 셋 저장
# df = pd.read_csv('Secom/data/df_IterativeImputer20.csv', index_col=0) # 결측치 처리된 데이터 셋 로드

# IsolationForest 하이퍼파라미터 최적화
def cv_silhouette_scorer(estimator, X): # 베이지안 최적화에 사용할 실루엣점수 정의
    estimator.fit(X)
    anomaly_scores = estimator.decision_function(X)
    cluster_labels = (anomaly_scores > np.quantile(anomaly_scores,0.003)).astype(int)
    num_labels = len(set(cluster_labels))
    num_samples = len(X)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return silhouette_score(X, cluster_labels)

df_sub = df.loc[:,~df.columns.isin(np.append(columns_nunique_1,'Pass/Fail'))]

opt = BayesSearchCV( # 베이지안 최적화 정의
        IsolationForest(bootstrap=False, contamination=0.01),
        {
        'n_estimators': (10,1000),
        'max_samples': (0.1,1)
        },
        n_iter=20,
        cv=5,
        scoring=cv_silhouette_scorer
)
opt.fit(df_sub.values)
print(opt.best_params_)

# IsolationForest 기반 이상치 처리
model = IsolationForest(
        bootstrap=False,
        contamination=0.01, 
        n_estimators=366,
        max_samples=0.96497204514265,
        random_state=999)
model.fit(df_sub.values)
df['outliers'] = model.predict(df_sub.values)

for col in df.columns[:-2]:
        df.loc[df['outliers']==-1,col]=np.nan
df = df.fillna(df.mean())

df = df.drop(columns='outliers')

#4.스케일링
df_X = df.iloc[:,:-1].copy()

scale = StandardScaler().fit(df_X)
df_X_std = scale.transform(df_X)
df_X_std = pd.DataFrame(df_X_std,columns=df_X.columns)
df_std = pd.concat([df_X_std,df.iloc[:,-1]],axis=1)

#5.특성 선택
# SelectKBest를 활용한 특성 선택 후보 탐색
X = df_std.drop(columns='Pass/Fail')
y = df_std['Pass/Fail'].replace(-1, 0)

all_columns = X.columns

sel = SelectKBest(score_func=f_classif,k=20)
sel_fit = sel.fit(X,y)
X = sel_fit.transform(X)

selected_mask = sel_fit.get_support()
print(all_columns[selected_mask])

X = df_std[all_columns[selected_mask]]

df_view = pd.DataFrame({'feature': list(all_columns[selected_mask]), 'score': list(sel.scores_[selected_mask])})
df_view = df_view.sort_values(by='score', ascending=False).reset_index(drop=True)
print(df_view)
sns.barplot(data=df_view,x='feature',y='score', legend=False)
for index, row in df_view.iterrows():
    plt.text(row.name, round(row.score,1), round(row.score,1), color='black', ha="center")
plt.xlabel('')
plt.ylabel('')
plt.show()

# RFE를 활용한 최종 특성 선택
model = RandomForestClassifier(
                    max_features='sqrt',
                    max_depth=15,
                    n_estimators=1000,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=999)

rfe = RFE(model, n_features_to_select=6)  # 선택할 특성의 개수를 지정합니다.

all_columns = X.columns

# RFE 알고리즘을 사용하여 특성 선택을 수행합니다.
rfe_fit = rfe.fit(X, y)

# 선택된 특성들을 출력합니다.
selected_mask = rfe_fit.support_
print(all_columns[selected_mask])
print(rfe_fit.estimator_.feature_importances_)
df_view = pd.DataFrame({'feature': list(all_columns[selected_mask]), 'score': list(rfe_fit.estimator_.feature_importances_)})
df_view = df_view.sort_values(by='score', ascending=False).reset_index(drop=True)
print(df_view)
sns.barplot(data=df_view,x='feature',y='score', legend=False, color='tab:orange')
for index, row in df_view.iterrows():
    plt.text(row.name, round(row.score,3), round(row.score,3), color='black', ha="center")
plt.xlabel('')
plt.ylabel('')
plt.show()

plt.figure(figsize=(8, 8))
sns.heatmap(df_std[all_columns[selected_mask]].corr(), annot=True, cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.show()

X = df_std[all_columns[selected_mask]]

#6.데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=101)

# 오버샘플링
smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=101)
X_train, y_train = smote.fit_resample(X_train, y_train)

#7.하이퍼파라미터 최적화
# 모델 정의
models_opt = {
    'LR': LogisticRegression(),
    'SVM': SVC(kernel='rbf',gamma='auto'),
    'RF': RandomForestClassifier(max_features='sqrt'),
    'XGB': XGBClassifier(),
    'LGBM': LGBMClassifier(verbose=-1)
}

# 하이퍼파라미터 탐색 범위 지정
search_space = {
    'LR': {
        'max_iter': (100, 1000),
        'C': (0.01, 100)
    },
    'SVM': {
        'C': (0.01, 100)
    },
    'RF': {
        'n_estimators': (10, 1000),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 10)
    },
    'XGB': {
        'n_estimators': (10, 1000),
        'max_depth': (3, 20),
        'min_child_weight': (1, 20),
        'learning_rate': (0.001, 1.0),
        'subsample': (0.1,1)
    },
    'LGBM': {
        'n_estimators': (10, 1000),
        'max_depth': (3, 20),
        'min_child_weight': (1, 20),
        'learning_rate': (0.001, 1.0),
        'num_leaves': (10,1000)
    }
}

# 베이지안 최적화를 통한 모델별 하이퍼파라미터 튜닝
best_params = {}
for model_name, model in models_opt.items():
    opt = BayesSearchCV(
        model,
        search_space[model_name],
        n_iter=10,
        cv=5,
        scoring='recall'
    )
    opt.fit(X_train, y_train)
    best_params[model_name] = opt.best_params_

# 최적화 결과 출력
for model_name, params in best_params.items():
    print(model_name, params)

#8.모델 학습 및 검증, 테스트
models = []

# LogisticRegression
model_lr = LogisticRegression(
                    max_iter=257,
                    C=33.40378818437279,
                    random_state=999)
models.append(('LR', model_lr))

# SVM
model_svm = SVC(kernel='rbf',
                    gamma='auto',
                    C=72.52471630372135,
                    random_state=999)
models.append(('SVM', model_svm))

# RandomForestClassifier
model_rf = RandomForestClassifier(
                    max_features='sqrt',
                    max_depth=16,
                    n_estimators=203,
                    min_samples_split=9,
                    min_samples_leaf=1,
                    random_state=999)
models.append(('RF', model_rf))

# XGBClassifier
model_xgb = XGBClassifier(
                    max_depth=10,
                    n_estimators=862,
                    min_child_weight=8,
                    learning_rate=0.14245197436308377,
                    subsample=0.9450130261919871,
                    random_state=999)
models.append(('XGB', model_xgb))

# LGBMClassifier
model_lgbm = LGBMClassifier(
                    max_depth=50,
                    n_estimators=257,
                    min_child_weight=8,
                    learning_rate=0.8833751829421289,
                    num_leaves=198,
                    verbose=-1,
                    random_state=999)
models.append(('LGBM', model_lgbm))

# 모델 평가
df_models_optimized = pd.DataFrame(columns=['Model_name','Accuracy','Recall'])
kfold = KFold(n_splits=5,shuffle=True,random_state=999)
for name, model in models:
    model_optimized = pd.DataFrame({
                    'Model_name':[name],
                    'Accuracy': [cross_validate(model, X_train, y_train, cv=kfold, scoring='accuracy')['test_score'].mean().round(3)],
                    'Recall':[cross_validate(model, X_train, y_train, cv=kfold, scoring='recall')['test_score'].mean().round(3)]
    })
    df_models_optimized = pd.concat([df_models_optimized,model_optimized], ignore_index=True)

# 결과 시각화
print(df_models_optimized)
df_models_optimized_melt = df_models_optimized.melt(id_vars=['Model_name'], var_name='variable', value_name='value')

fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x='Model_name', y='value', hue='variable', data=df_models_optimized_melt, palette=['#80A8E5','#A06CE4'])
for p in ax.patches:
    if p.get_width()==0 and p.get_height()==0:
        break
    ax.text(p.get_x() + (p.get_width()/2),
            p.get_y() + p.get_height(),
            p.get_height(),
            ha = 'center' )
plt.ylim(0, 1)
plt.xlabel('')
plt.ylabel('')
plt.legend(title=None)
plt.show()

# 테스트
model = XGBClassifier(
                    max_depth=10,
                    n_estimators=862,
                    min_child_weight=8,
                    learning_rate=0.14245197436308377,
                    subsample=0.9450130261919871,
                    random_state=999)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

# 정확도 평가
result = pd.Series(y_predict, index=y_test.index)

accuracy = accuracy_score(y_test, result)
print(f'Test Accuracy Score: {accuracy:.3f}')

# 혼동행렬 시각화
cm = confusion_matrix(y_test, result)
labels = ['Negative', 'Positive']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()