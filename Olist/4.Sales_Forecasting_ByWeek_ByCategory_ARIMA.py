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
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

#1.데이터 로드
df_orders = pd.read_csv('./data/olist_orders_dataset.csv')
df_items = pd.read_csv('./data/olist_order_items_dataset.csv')
df_products = pd.read_csv('./data/olist_products_dataset.csv')
df_product_category_name = pd.read_csv('./data/product_category_name_translation.csv')

# df_orders 전처리
df_orders = df_orders[['order_id','order_purchase_timestamp']]

# df_items 전처리 및 merge
df_items_grouped = df_items.groupby(['order_id','product_id'],as_index=False).agg(
    price = ('price','sum'))
df = pd.merge(df_orders,df_items_grouped,on='order_id', how='left')

# df_products 전처리 및 merge
df_products = df_products[['product_id','product_category_name']]
df = pd.merge(df,df_products,on='product_id',how='left')

# df_product_category_name merge
df = pd.merge(df,df_product_category_name,on='product_category_name',how='left')

# 구매일자 날짜 형식 변환(timestamp에서 date로)
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], format='%Y-%m-%d %H:%M:%S')
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].dt.date
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], format='%Y-%m-%d')

# week별 product_category에 대한 sales 합계 구하기
df = df.set_index('order_purchase_timestamp')
df.rename_axis('date', inplace=True)
df = df.groupby('product_category_name_english').resample('W').agg(sales=('price','sum'))
df = df.sort_index()
df = df.reset_index()
df = df.set_index('date')
df = df.rename(columns={'product_category_name_english': 'category'})

#2.EDA
# 카테고리별 판매실적 상위 10개
sales_by_category = df.groupby('category',as_index=False)['sales'].sum()
sales_by_category = sales_by_category.sort_values(by='sales',ascending=False).iloc[0:10]
print(sales_by_category['category'])

# 상위 10개에 속하는 카테고리에 대한 단기/장기 이동평균 비율
short_rolling = df[df['category'].isin(sales_by_category['category'])].groupby('category')['sales'].rolling(window=4).mean()
long_rolling = df[df['category'].isin(sales_by_category['category'])].groupby('category')['sales'].rolling(window=26).mean()
short_long_rolling_ratio = pd.DataFrame((short_rolling/long_rolling))
short_long_rolling_ratio = short_long_rolling_ratio.reset_index()
short_long_rolling_ratio_by_category = short_long_rolling_ratio.groupby('category',as_index=False).agg(sales = ('sales','mean'))
short_long_rolling_ratio_by_category = short_long_rolling_ratio_by_category.sort_values(by='sales',ascending=False)

# 시각화
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.bar(sales_by_category['category'], sales_by_category['sales'], color='tab:gray', label='Total sales')
ax2 = ax1.twinx()
ax2.plot(short_long_rolling_ratio_by_category['category'], short_long_rolling_ratio_by_category['sales'], color='tab:orange', marker='o', linestyle='None', label='Rolling ratio')

# 각 축의 범례 핸들과 라벨을 가져와서 결합
ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
ax1.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels, loc='upper right')
plt.show()

# Rolling ratio 상위 3개 시각화
Rolling_ratio_top3_category = ['health_beauty','watches_gifts','housewares']
df_view = df[df['category'].isin(Rolling_ratio_top3_category)]

plt.figure(figsize=(20,10))
sns.lineplot(data=df_view,x=df_view.index,hue='category',hue_order=sales_by_category[sales_by_category['category'].isin(Rolling_ratio_top3_category)].groupby('category')['sales'].sum().sort_values(ascending=False).index.tolist(),y='sales')
plt.show()

# Rolling ratio 하위 3개 시각화
Rolling_ratio_bot3_category = ['sports_leisure','cool_stuff','garden_tools']
df_view = df[df['category'].isin(Rolling_ratio_bot3_category)]

plt.figure(figsize=(20,10))
sns.lineplot(data=df_view,x=df_view.index,hue='category',hue_order=sales_by_category[sales_by_category['category'].isin(Rolling_ratio_bot3_category)].groupby('category')['sales'].sum().sort_values(ascending=False).index.tolist(),y='sales')
plt.show()

#3.데이터 정제
# category 선택
sel_category = 'health_beauty'
df = df[df['category']==sel_category]
df = df.drop(columns='category')

plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y='sales')
plt.show()

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

# 계절성 확인
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
adf_asis = adfuller(df['sales'])
print('\n[AS-IS]ADF Statistic: %f' %adf_asis[0])
print('p-value: %f' %adf_asis[1])
print('Critical Values:', adf_asis[4])

adf_tobe = adfuller(df_diff['sales'])
print('[TO-BE]ADF Statistic: %f' %adf_tobe[0])
print('p-value: %f' %adf_tobe[1])
print('Critical Values:', adf_tobe[4])

kpss_stat_asis, p_value_asis, lags_asis, critical_values_asis = kpss(df['sales'])
print('\n[AS-IS]KPSS Statistic: %f' %kpss_stat_asis)
print('p-value: %f' %p_value_asis)
print('Critical Values:', critical_values_asis)

kpss_stat_tobe, p_value_tobe, lags_tobe, critical_values_tobe = kpss(df_diff['sales'])
print('[TO-BE]KPSS Statistic: %f' %kpss_stat_tobe)
print('p-value: %f' %p_value_tobe)
print('Critical Values:', critical_values_tobe)

fig, ax = plt.subplots(1,2,figsize=(10,5))

result_adf = pd.DataFrame({
    'div': ['AS-IS', 'TO-BE'],
    'p_value': [adf_asis[1], adf_tobe[1]]
})
sns.barplot(data=result_adf, x='div', y='p_value', hue='div', ax=ax[0], palette='tab10')
for index, row in result_adf.iterrows():
    ax[0].text(row.name, round(row.p_value,3), round(row.p_value,3), color='black', ha="center")
ax[0].set_title('ADF')
ax[0].set_xlabel('')
ax[0].set_ylabel('')

result_kpss = pd.DataFrame({
    'div': ['AS-IS', 'TO-BE'],
    'p_value': [p_value_asis, p_value_tobe]
})
sns.barplot(data=result_kpss, x='div', y='p_value', hue='div', ax=ax[1], palette='tab10')
for index, row in result_kpss.iterrows():
    ax[1].text(row.name, round(row.p_value,3), round(row.p_value,3), color='black', ha="center")
ax[1].set_title('KPSS')
ax[1].set_xlabel('') 
ax[1].set_ylabel('') 
plt.show()

# 차분 전후 ACF 및 PACF 지표 확인(모델 초기 파라미터 판단)
fig, ax = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('Autocorrelation')
sgt.plot_acf(df['sales'],ax=ax[0])
sgt.plot_acf(df_diff['sales'],ax=ax[1])
ax[0].set_title('AS-IS')
ax[1].set_title('TO-BE')
plt.show()

fig, ax = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('Partial Autocorrelation')
sgt.plot_pacf(df['sales'],ax=ax[0])
sgt.plot_pacf(df_diff['sales'],ax=ax[1])
ax[0].set_title('AS-IS')
ax[1].set_title('TO-BE')
plt.show()

# train/test 데이터 분리
df_train = df[(df.index>=np.datetime64('2017-02-01', 'D'))&(df.index<np.datetime64('2018-02-01', 'D'))]
df_test = df[(df.index>=np.datetime64('2018-02-01', 'D'))&(df.index<np.datetime64('2018-08-01', 'D'))]

#4.파라미터 최적화
# 최적의 파라미터 탐색
pdq = list(itertools.product(range(0,3),range(1,2),range(0,3)))
seasonal_pdq = [(x[0],x[1],x[2],frequency) for x in list(itertools.product(range(0,3),range(0,2),range(0,3)))]
print(pdq)
print(seasonal_pdq)

aic = []
mae = []
params = []
n_steps = len(df_test)

for i in pdq:
    for j in seasonal_pdq:
        try:
            model = ARIMA(df_train, order=i, seasonal_order=j)
            model_fit = model.fit()

            get_forecast = model_fit.get_forecast(steps=n_steps)
            summary_frame = get_forecast.summary_frame(alpha=0.05)
            forecast = summary_frame['mean']
            
            aic.append(model_fit.aic)
            mae.append(mean_absolute_error(df_test['sales'], forecast))
            params.append([i,j])
            
            print(f"{i}{j} AIC: {model_fit.aic}, MAE: {mean_absolute_error(df_test['sales'], forecast)}")

        except:
            continue

# 결과 출력
for i in range(len(params)):
    print(f"{params[i]} AIC:{aic[i]}, MAE:{mae[i]}")

print(f"\nbest >> {params[mae.index(min(mae))]} AIC:{aic[mae.index(min(mae))]}, MAE:{min(mae)}")

best_order = params[mae.index(min(mae))][0]
best_seasonal_order = params[mae.index(min(mae))][1]

# 결과 시각화
plt.scatter(x=aic, y=mae, color='tab:blue')
plt.scatter(aic[mae.index(min(mae))], min(mae), color='tab:orange')
plt.xlabel('AIC')
plt.ylabel('MAE')
plt.show()

best_model = pd.DataFrame({
    'div': ['AIC', 'MAE'],
    'value': [aic[mae.index(min(mae))], min(mae)]
})
sns.barplot(data=best_model, x='div', y='value', hue='div', palette=['#80A8E5','#A06CE4'])
for index, row in best_model.iterrows():
    plt.text(row.name, round(row.value,3), round(row.value,3), color='black', ha="center")
plt.xlabel('')
plt.ylabel('')
plt.show()

#5.모델 구축 및 평가
# ARIMA 모델 구축
model = ARIMA(df_train, order=best_order, seasonal_order=best_seasonal_order)
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
ax.vlines(forecast.index.min(), 0, pd.concat([df_train['sales'],df_test['sales']]).max()*1.2, linestyle='--', color='tab:red', label='Start of Prediction')
ax.legend(loc='upper left')
ax.set_title(sel_category)

# 정확도 평가(MAE)
mae = mean_absolute_error(df_test['sales'], forecast)
fig.text(0.867, 0.855, f"MAE: {round(mae,2)}", fontsize=6, ha='center')
# plt.savefig(f'./result/arima/back_{sel_category}.png')
plt.show()

# 모델 적합성 평가(AIC)
print(model_fit.summary()) 

fig, ax = plt.subplots(figsize=(10,5))
sgt.plot_acf(model_fit.resid,ax=ax) # 신뢰구간 내 자기상관이 존재하므로 적합성 양호
plt.show()

#6.실데이터 예측
# ARIMA 모델 구축
model = ARIMA(pd.concat([df_train,df_test]), order=best_order, seasonal_order=best_seasonal_order)
model_fit = model.fit()

# 예측 수행
n_steps = 13
get_forecast = model_fit.get_forecast(steps=n_steps)
summary_frame = get_forecast.summary_frame(alpha=0.05)
forecast = summary_frame['mean']

# 신뢰 구간 계산
ci_upper = summary_frame['mean_ci_upper']
ci_lower = summary_frame['mean_ci_lower']
ci_index = summary_frame.index

# 예측 시각화
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(pd.concat([pd.concat([df_train['sales'],df_test['sales']]),forecast[forecast.index==forecast.index.min()]]), label='Actual', color='tab:blue')
ax.plot(forecast, label='Forecast', color='tab:orange')
ax.fill_between(ci_index, ci_lower, ci_upper, color='tab:gray', alpha=0.1, label='95% Prediction Interval')
ax.vlines(forecast.index.min(), 0, pd.concat([df_train['sales'],df_test['sales']]).max()*1.2, linestyle='--', color='tab:red', label='Start of Prediction')
ax.plot(pd.concat([pd.concat([df_train['sales'],df_test['sales']]),forecast]).rolling(window=26).mean(), label='MA26',color='tab:red')
ax.legend(loc='upper left')
ax.set_title(sel_category)

# plt.savefig(f'./result/arima/{sel_category}.png')
plt.show()