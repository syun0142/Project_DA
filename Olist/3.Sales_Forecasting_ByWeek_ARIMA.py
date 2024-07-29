#0.라이브러리
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.graphics.tsaplots as sgt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

#1.데이터 로드
df_orders = pd.read_csv('Olist/data/olist_orders_dataset.csv')
df_items = pd.read_csv('Olist/data/olist_order_items_dataset.csv')

df_items_grouped = df_items.groupby(['order_id','product_id'],as_index=False).agg(
    price = ('price','sum'))
df = pd.merge(df_orders,df_items_grouped,on='order_id', how='left')

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], format='%Y-%m-%d %H:%M:%S')
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].dt.date
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

df = df.groupby(['order_purchase_timestamp']).agg(sales=('price','sum'))
df.rename_axis('date', inplace=True)
df = df.resample('W').sum()
df = df.sort_index()

#2.EDA
plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y='sales')
plt.show()

#3.데이터 정제
# 이상치 제거
df = df[(df.index>=np.datetime64('2017-02-01', 'D'))&(df.index<np.datetime64('2018-08-01', 'D'))]
df.loc['2017-11-26','sales'] = pd.NA

# 캘린더를 이용한 결측값 테스트
date_range = pd.date_range(start=df.index.min(), end=df.index.max())
calendar_data = pd.DataFrame(date_range, columns=['date'])
calendar_data = calendar_data.set_index('date')
calendar_data = calendar_data.resample('W').sum()
df_none_test = pd.merge(calendar_data,df,on='date',how='left')
print(df_none_test[df_none_test['sales'].isna()])

# 제거된 이상치 선형 보간법으로 대체
df['sales'] = df['sales'].interpolate(method='linear')

# 주기 설정
df.index.freq = 'W'
frequency = 13

# 시계열 분해
decomposition = sm.tsa.seasonal_decompose(df['sales'],model='additive',period=frequency)
fig = decomposition.plot()
fig.set_size_inches(10, 5)
plt.show()

# 잔차의 정규성 확인(로그 변환 불필요하다고 판단)
sm.qqplot(decomposition.resid, line='s')
plt.title('QQ Plot of Residuals')
plt.show()

# 계절성 확인(11~13w 중에 가장 성능이 좋은 13w 채택)
frequencies, power_spectrum = periodogram(df['sales'])
plt.plot(frequencies, power_spectrum)
plt.title('Periodogram')
plt.xlabel('Frequency')
plt.ylabel('Power Spectrum')
plt.grid(True, alpha=0.2)
plt.show()

# 차분 테스트
df_diff = df.diff().dropna()

# 차분 전후 데이터 정상성 테스트
result = adfuller(df['sales'])
print('\n[AS-IS]ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:', result[4])

result = adfuller(df_diff['sales'])
print('[TO-BE]ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:', result[4])

kpss_stat, p_value, lags, critical_values = kpss(df['sales'])
print('\n[AS-IS]KPSS Statistic: %f' % kpss_stat)
print('p-value: %f' % p_value)
print('Critical Values:', critical_values)

kpss_stat, p_value, lags, critical_values = kpss(df_diff['sales'])
print('[TO-BE]KPSS Statistic: %f' % kpss_stat)
print('p-value: %f' % p_value)
print('Critical Values:', critical_values)

# 차분 전후 ACF 및 PACF 지표 확인(모델 초기 파라미터 판단)
fig, ax = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('AS-IS')
sgt.plot_acf(df['sales'],ax=ax[0])
sgt.plot_pacf(df['sales'],ax=ax[1])
plt.show()

fig, ax = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('TO-BE')
sgt.plot_acf(df_diff['sales'],ax=ax[0])
sgt.plot_pacf(df_diff['sales'],ax=ax[1])
plt.show()

# train/test 데이터 분리
df_train = df[(df.index>=np.datetime64('2017-02-01', 'D'))&(df.index<np.datetime64('2018-02-01', 'D'))]
df_test = df[(df.index>=np.datetime64('2018-02-01', 'D'))&(df.index<np.datetime64('2018-08-01', 'D'))]

#4.파라미터 최적화
# auto_arima
auto_arima_model = auto_arima(df_train, max_p=3, max_q=3, max_P=3, max_Q=3, m=frequency, d=1, D=1, seasonal=True, trace=True, stepwise=False)
print(auto_arima_model.summary())

#5.모델 구축 및 평가
# ARIMA 모델 구축
model = ARIMA(df_train, order=(1, 1, 1), seasonal_order=(0, 0, 2, frequency))
model_fit = model.fit()

# 예측 수행
n_steps = len(df_test)
get_forecast = model_fit.get_forecast(steps=n_steps)
summary_frame = get_forecast.summary_frame(alpha=0.05)
forecast = summary_frame['mean']

# 신뢰 구간 계산
ci_upper = summary_frame['mean_ci_upper']
ci_lower = summary_frame['mean_ci_lower']
ci_index = summary_frame.index

# 예측 시각화
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(pd.concat([df_train['sales'],df_test['sales']]), label='Actual', color='tab:blue')
ax.plot(forecast, label='Forecast', color='tab:orange')
ax.fill_between(ci_index, ci_lower, ci_upper, color='tab:gray', alpha=0.1, label='95% Prediction Interval')
ax.vlines('2018-02-01', 0, pd.concat([df_train['sales'],df_test['sales']]).max()*1.2, linestyle='--', color='r', label='Start of Prediction')
ax.legend(loc='upper left')
plt.show()

# 모델 적합성 평가(AIC)
print(model_fit.summary()) 

fig, ax = plt.subplots(figsize=(10,5))
sgt.plot_acf(model_fit.resid,ax=ax) # 신뢰구간 내 자기상관이 존재하므로 적합성 양호
plt.show()

# 정확도 평가
mae = mean_absolute_error(df_test['sales'], forecast.loc[:df_test.index.max()])
print("Mean Absolute Error (MAE): %.3f" % mae)