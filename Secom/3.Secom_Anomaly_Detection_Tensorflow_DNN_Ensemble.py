#0.라이브러리
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.ensemble import AdaBoostClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score

#1.데이터 로드
df = pd.read_csv('Secom/data/uci-secom.csv')

#2.EDA
print(df.info())
print(df)
print(df.select_dtypes(exclude='float'))
print(df['Pass/Fail'].value_counts())
sns.scatterplot(data=df,x='Time',y='Pass/Fail') # Pass/Fail 분포 확인
plt.xticks([])
plt.show()

# Pass/Fail과 독립변수에 대한 간단한 상관관계 분석
print(df.drop(columns='Time').corr()['Pass/Fail'].abs().sort_values(ascending=False).head(11))
corr_matrix = df[['59','103','510','348','158','431','Pass/Fail']].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, linewidths=0.5)
plt.show()

sns.pairplot(df[['59','103','510','348','158','431','Pass/Fail']])
plt.show()

#3.데이터 정제
# 불필요한 컬럼 제거
df = df.drop(columns='Time')

# 고유값이 1인 특성에 대해 NaN를 -1로 대체
# print((df.loc[:,df.nunique()==1].sum()<0).sort_values(ascending=False))
columns_nunique_1 = df.loc[:,df.nunique()==1].columns
for col in columns_nunique_1:
        df.loc[df[col].isna(),col]=-1

# 결측치 많은 컬럼 제거
# print(df.isna().sum().sort_values(ascending=False).head(60))
df = df.loc[:,(df.isna().sum()>len(df)*0.4)==False]

# 결측치 처리
df = df.fillna(df.mean())

# 사분위수 기반 이상치 처리
df_sub = df.loc[:,~df.columns.isin(np.append(columns_nunique_1,'Pass/Fail'))]

for col in df_sub.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3-Q1
        df.loc[(df[col]<(Q1-1.5*IQR)) | (df[col]>(Q3+1.5*IQR)),col]=np.NaN
df = df.fillna(df.mean())

#4.스케일링
df_X = df.iloc[:,:-1].copy()

scale = StandardScaler().fit(df_X)
df_X_std = scale.transform(df_X)
df_X_std = pd.DataFrame(df_X_std,columns=df_X.columns)
df_std = pd.concat([df_X_std,df.iloc[:,-1]],axis=1)

#5.데이터 분리
X = df_std.drop(columns='Pass/Fail')
y = df_std['Pass/Fail'].replace(-1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
X1_train, X2_train, y1_train, y2_train = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

#6.하이퍼파라미터 최적화
random.seed(999)
tf.random.set_seed(999)

# 함수 형태로 모델 생성
def create_model(n_hidden_layers, layer1_units, layer2_units, layer3_units, learning_rate, layer1_activation, layer2_activation, layer3_activation):

    created_model = Sequential()
    created_model.add(Input(shape=(X_train.shape[1],)))

    layer_units = [layer1_units, layer2_units, layer3_units]
    layer_activation = [layer1_activation, layer2_activation, layer3_activation]
    
    for i in range(n_hidden_layers):
        created_model.add(Dense(units=layer_units[i], activation='relu' if layer_activation[i] < 0.5 else 'sigmoid'))
        
    created_model.add(Dense(1, activation='sigmoid'))
    
    created_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return created_model

# BayesianOptimization을 사용하여 하이퍼파라미터 최적화
def optimize_hyperparameters(n_estimators, n_hidden_layers, layer1_units, layer2_units, layer3_units, learning_rate, epochs, batch_size, layer1_activation, layer2_activation, layer3_activation):
    model_KerasClassifier = KerasClassifier(model=create_model, 
                            n_hidden_layers=int(n_hidden_layers),
                            layer1_units=int(layer1_units),
                            layer2_units=int(layer2_units),
                            layer3_units=int(layer3_units),
                            learning_rate=learning_rate,
                            layer1_activation=layer1_activation,
                            layer2_activation=layer2_activation,
                            layer3_activation=layer3_activation,
                            epochs=int(epochs), 
                            batch_size=int(batch_size), 
                            validation_split=0.2,
                            verbose=0)
    
    model_ada_boost = AdaBoostClassifier(estimator=model_KerasClassifier, n_estimators=int(n_estimators), algorithm='SAMME', random_state=999)
    model_ada_boost.fit(X1_train,y1_train)
    
    accuracy = accuracy_score(y2_train, model_ada_boost.predict(X2_train))
    
    return accuracy

opt = BayesianOptimization(
    f=optimize_hyperparameters,
    pbounds={'n_estimators': (10, 100),
             'n_hidden_layers': (1, 3),
             'layer1_units': (32, 128),
             'layer2_units': (32, 128),
             'layer3_units': (32, 128),
             'learning_rate': (0.0001, 0.01),
             'epochs':(2,50),
             'batch_size':(16,128),
             'layer1_activation': (0, 1),
             'layer2_activation': (0, 1),
             'layer3_activation': (0, 1)
            }
)

opt.maximize(init_points=5, n_iter=10)

print("Best Params:", opt.max['params'])
print("Best Accuracy:", opt.max['target'])

#7.딥러닝(자동)
# 난수 시드 설정
random.seed(999)
tf.random.set_seed(999)

# 모델 생성 (DNN)
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))

layer_units = [opt.max['params']['layer1_units'], opt.max['params']['layer2_units'], opt.max['params']['layer3_units']]
layer_activation = [opt.max['params']['layer1_activation'], opt.max['params']['layer2_activation'], opt.max['params']['layer3_activation']]

for i in range(int(opt.max['params']['n_hidden_layers'])):
    model.add(Dense(units=int(layer_units[i]), activation='relu' if layer_activation[i] < 0.5 else 'sigmoid'))
    
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=opt.max['params']['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습 및 검증 (AdaBoost 결합)
model_KerasClassifier = KerasClassifier(model=model, epochs=int(opt.max['params']['epochs']), batch_size=int(opt.max['params']['batch_size']), validation_split=0.2, verbose=0)
model_ada_boost = AdaBoostClassifier(estimator=model_KerasClassifier, n_estimators=int(opt.max['params']['n_estimators']), algorithm='SAMME', random_state=999)
model_ada_boost.fit(X_train, y_train)

# 테스트 정확도 평가
accuracy = accuracy_score(y_test, model_ada_boost.predict(X_test))
print(f'Test Accuracy Score: {accuracy:.3f}')