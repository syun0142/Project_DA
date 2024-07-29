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
from tensorflow.keras.optimizers import Adam # type: ignore
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix

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

# 평균으로 결측치 처리
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

#6.하이퍼파라미터 최적화
# 난수 시드 설정
random.seed(999)
tf.random.set_seed(999)

# 베이지안 최적화 모델로부터 파라미터를 입력받아 DNN 모델의 정확도를 반환하는 함수 정의
def create_model(learning_rate, n_hidden_layers, layer1_units, layer2_units, layer3_units, layer1_activation, layer2_activation, layer3_activation, epochs, batch_size):
    created_model = Sequential()
    created_model.add(Input(shape=(len(X_train.columns),)))

    layer_units = [layer1_units, layer2_units, layer3_units]
    layer_activation = [layer1_activation, layer2_activation, layer3_activation]
    
    for i in range(int(n_hidden_layers)):
        created_model.add(Dense(units=int(layer_units[i]), activation='relu' if layer_activation[i] < 0.5 else 'sigmoid'))
    
    created_model.add(Dense(1, activation='sigmoid'))
    
    created_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    created_model_fit = created_model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.2, verbose=0)

    accuracy = created_model_fit.history['val_accuracy'][-1]
    
    return accuracy

opt = BayesianOptimization( # 베이지안 최적화 정의
    f=create_model,
    pbounds={
        'learning_rate': (0.0001, 0.01),
        'n_hidden_layers': (1, 3),
        'layer1_units': (32, 256),
        'layer2_units': (32, 256),
        'layer3_units': (32, 256),
        'layer1_activation': (0, 1),
        'layer2_activation': (0, 1),
        'layer3_activation': (0, 1),
        'epochs':(2,50),
        'batch_size':(16,128),
    }
)

opt.maximize(init_points=5, n_iter=10)

print("Best Params:", opt.max['params'])
print("Best Accuracy:", opt.max['target'])

#7.딥러닝
# 난수 시드 설정
random.seed(999)
tf.random.set_seed(999)

# 모델 생성 (DNN)
model = Sequential([
    Input(shape=(len(X_train.columns),)),
    Dense(212, activation='sigmoid'),
    Dense(140, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.008358586518712194), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습 및 검증
model_fit = model.fit(X_train, y_train, epochs=4, batch_size=58, validation_split=0.2)

# 모델 학습 및 검증 결과 시각화
history = pd.DataFrame(model_fit.history)

for col1, col2 in [('accuracy','loss'),('val_accuracy','val_loss')]:
    df_view = history.iloc[-1][[col1,col2]].reset_index().rename(columns={'index':'div',3:'value'})

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x='div', y='value', data=df_view, palette=['#80A8E5','#A06CE4'])
    for p in ax.patches:
        if p.get_width()==0 and p.get_height()==0:
            break
        ax.text(p.get_x() + (p.get_width()/2),
                p.get_y() + p.get_height(),
                round(p.get_height(),3),
                ha = 'center' )
    plt.ylim(0, 1)
    plt.xlabel('')
    plt.ylabel('')
    ax.legend().remove()
    plt.show()

# 테스트 정확도 평가
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy Score: {accuracy:.3f}')

# 교차행렬 시각화
y_predict_prob = model.predict(X_test).ravel()
y_predict = (y_predict_prob > 0.5).astype(int)
result = pd.Series(y_predict, index=y_test.index)

cm = confusion_matrix(y_test, y_predict)
labels = ['Positive', 'Negative']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()