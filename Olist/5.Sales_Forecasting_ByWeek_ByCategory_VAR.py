#0.라이브러리
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import statsmodels.graphics.tsaplots as sgt

#1.데이터 로드
df_orders = pd.read_csv('Olist/data/olist_orders_dataset.csv')
df_reviews = pd.read_csv('Olist/data/olist_order_reviews_dataset.csv')
df_items = pd.read_csv('Olist/data/olist_order_items_dataset.csv')
df_products = pd.read_csv('Olist/data/olist_products_dataset.csv')
df_product_category_name = pd.read_csv('Olist/data/product_category_name_translation.csv')

# df_orders 전처리
df_orders = df_orders[['order_id','order_purchase_timestamp']]

# df_reviews 전처리 및 merge
df_reviews_grouped = df_reviews.groupby(['order_id'],as_index=False)['review_score'].mean()
df = pd.merge(df_orders,df_reviews_grouped,on='order_id',how='left')

# df_items 전처리 및 merge
df_items_grouped = df_items.groupby(['order_id','product_id'],as_index=False).agg(
    price = ('price','sum'))
df = pd.merge(df,df_items_grouped,on='order_id', how='left')

# df_products 전처리 및 merge
df_products = df_products[['product_id','product_category_name']]
df = pd.merge(df,df_products,on='product_id',how='left')

# df_product_category_name merge
df = pd.merge(df,df_product_category_name,on='product_category_name',how='left')

# 구매일자 날짜 형식 변환(timestamp에서 date로)
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], format='%Y-%m-%d %H:%M:%S')
df['order_purchase_timestamp'] = df['order_purchase_timestamp'].dt.date
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], format='%Y-%m-%d')

# week별 product_category에 대한 sales 합계, review_score 평균 구하기
df = df.set_index('order_purchase_timestamp')
df.rename_axis('date', inplace=True)
df = df.groupby('product_category_name_english').resample('W').agg(
    sales=('price','sum')
    ,review_score=('review_score','mean')
)
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

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(df.index, df['sales'], color='tab:blue', label='Sales')
ax2 = ax1.twinx()
ax2.plot(df.index, df['review_score'], color='tab:green', label='Review Score')

# 각 축의 범례 핸들과 라벨을 가져와서 결합
ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
ax1.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels, loc='upper left')
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
print(df_none_test[df_none_test['review_score'].isna()])

# 제거된 이상치 선형 보간법으로 대체
df['sales'] = df['sales'].interpolate(method='linear')
df['review_score'] = df['review_score'].interpolate(method='linear')

# 주기 설정
df.index.freq = 'W'
frequency = 13

## 특성 확인
for col in ['sales','review_score']:
    print('\n############## %s 특성 확인 ##############' %col)

    # 시계열 분해
    decomposition = sm.tsa.seasonal_decompose(df[col],model='additive',period=frequency)
    fig = decomposition.plot()
    fig.set_size_inches(10, 5)
    plt.show()
    
    # 잔차의 정규성 확인(로그 변환 불필요하다고 판단)
    sm.qqplot(decomposition.resid, line='s')
    plt.title('QQ Plot of Residuals')
    plt.show()
    
    # 계절성 확인
    frequencies, power_spectrum = periodogram(df[col])
    plt.plot(frequencies, power_spectrum)
    plt.title('Periodogram')
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectrum')
    plt.grid(True, alpha=0.2)
    plt.show()
    
    # 차분 테스트
    df_diff = pd.DataFrame(df[col].diff().dropna())
    
    # 차분 전후 데이터 정상성 테스트
    adf_asis = adfuller(df[col])
    print('\n[AS-IS]ADF Statistic: %f' %adf_asis[0])
    print('p-value: %f' %adf_asis[1])
    print('Critical Values:', adf_asis[4])

    adf_tobe = adfuller(df_diff[col])
    print('[TO-BE]ADF Statistic: %f' %adf_tobe[0])
    print('p-value: %f' %adf_tobe[1])
    print('Critical Values:', adf_tobe[4])

    kpss_stat_asis, p_value_asis, lags_asis, critical_values_asis = kpss(df[col])
    print('\n[AS-IS]KPSS Statistic: %f' %kpss_stat_asis)
    print('p-value: %f' %p_value_asis)
    print('Critical Values:', critical_values_asis)

    kpss_stat_tobe, p_value_tobe, lags_tobe, critical_values_tobe = kpss(df_diff[col])
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
        ax[1].text(row.name, row.p_value, row.p_value, color='black', ha="center")
    ax[1].set_title('KPSS')
    ax[1].set_xlabel('') 
    ax[1].set_ylabel('') 
    plt.show()

    # 차분 전후 ACF 및 PACF 지표 확인(모델 초기 파라미터 판단)
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle('Autocorrelation')
    sgt.plot_acf(df[col],ax=ax[0])
    sgt.plot_acf(df_diff[col],ax=ax[1])
    ax[0].set_title('AS-IS')
    ax[1].set_title('TO-BE')
    plt.show()

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle('Partial Autocorrelation')
    sgt.plot_pacf(df[col],ax=ax[0])
    sgt.plot_pacf(df_diff[col],ax=ax[1])
    ax[0].set_title('AS-IS')
    ax[1].set_title('TO-BE')
    plt.show()

# train/test 데이터 분리
df_train = df[(df.index>=np.datetime64('2017-02-01', 'D'))&(df.index<np.datetime64('2018-02-01', 'D'))]
df_test = df[(df.index>=np.datetime64('2018-02-01', 'D'))&(df.index<np.datetime64('2018-08-01', 'D'))]

# sales 차분 수행 후 train/test 데이터 분리
df_diff = df.copy()
df_diff['sales'] = df_diff['sales'].diff().dropna()
df_diff = df_diff.dropna()

df_diff_train = df_diff[(df_diff.index>=np.datetime64('2017-02-01', 'D'))&(df_diff.index<np.datetime64('2018-02-01', 'D'))]
df_diff_test = df_diff[(df_diff.index>=np.datetime64('2018-02-01', 'D'))&(df_diff.index<np.datetime64('2018-08-01', 'D'))]

#4.파라미터 최적화
# AIC을 사용하여 최적의 lag 수 선택
best_sales_mae = float('inf')
best_review_score_mae = float('inf')
best_lag = 0
lag_list = []
sales_mae_list = []
review_score_mae_list = []

# 최적의 lag 수 찾기
for l in range(1, int(len(df_diff_train)/3)):
    model = VAR(df_diff_train)
    model_fit = model.fit(l)

    forecast_steps = len(df_diff_test)
    forecast_interval = model_fit.forecast_interval(df_diff_train.values,steps=forecast_steps, alpha=0.05)
    forecast = pd.DataFrame(forecast_interval[0], columns=['sales', 'review_score'])
    forecast.index = pd.date_range(start=df_diff_train.index.max() + timedelta(days=7), periods=forecast_steps, freq='W')
    forecast = forecast.rename_axis("date")
    forecast['sales'] = forecast['sales'].round(2)

    # forecast 역차분
    forecast.loc[forecast.index.min(),'sales'] += df_train['sales'].iloc[-1]
    forecast['sales'] = forecast['sales'].cumsum()

    aic = model_fit.aic
    sales_mae = mean_absolute_error(df_test['sales'], forecast.loc[:df_test.index.max(),'sales'])
    review_score_mae = mean_absolute_error(df_test['review_score'], forecast.loc[:df_test.index.max(),'review_score'])

    lag_list.append(l)
    sales_mae_list.append(sales_mae)
    review_score_mae_list.append(review_score_mae)
    print(f"[lag {l}] AIC: {aic}, sales MAE: {sales_mae}, review score MAE: {review_score_mae}")

    if (sales_mae*review_score_mae) < (best_sales_mae*best_review_score_mae):
        best_sales_mae = sales_mae
        best_review_score_mae = review_score_mae
        best_lag = l

print(f"MAE best: {best_lag}")

# 결과 시각화
plt.scatter(x=sales_mae_list, y=review_score_mae_list, color='tab:blue')
plt.scatter(sales_mae_list[lag_list.index(best_lag)], review_score_mae_list[lag_list.index(best_lag)], color='tab:orange')
plt.xlabel('Sales MAE')
plt.ylabel('Review Score MAE')
plt.show()

best_model = pd.DataFrame({
    'div': ['Sales MAE', 'Review Score MAE'],
    'value': [sales_mae_list[lag_list.index(best_lag)], review_score_mae_list[lag_list.index(best_lag)]]
})
sns.barplot(data=best_model, x='div', y='value', hue='div', palette=['#80A8E5','#A06CE4'])
for index, row in best_model.iterrows():
    plt.text(row.name, round(row.value,3), round(row.value,3), color='black', ha="center")
plt.xlabel('')
plt.ylabel('')
plt.show()

#5.모델 구축 및 평가
# VAR 모델 구축
model = VAR(df_diff_train)
model_fit = model.fit(best_lag)

# 예측 수행
forecast_steps = len(df_diff_test)
forecast_interval = model_fit.forecast_interval(df_diff_train.values,steps=forecast_steps, alpha=0.05)
forecast = pd.DataFrame(forecast_interval[0], columns=['sales', 'review_score'])
forecast.index = pd.date_range(start=df_diff_train.index.max() + timedelta(days=7), periods=forecast_steps, freq='W')
forecast = forecast.rename_axis("date")
forecast['sales'] = forecast['sales'].round(2)

# forecast 역차분
forecast.loc[forecast.index.min(),'sales'] += df_train['sales'].iloc[-1]
forecast['sales'] = forecast['sales'].cumsum()

# 예측 시각화
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(pd.concat([df_train['sales'],df_test['sales']]), label='Sales(Actual)', color='tab:blue')
ax1.plot(forecast['sales'], label='Sales(Fcst)', color='tab:orange')
ax2 = ax1.twinx()
ax2.plot(forecast['review_score'], label='Review Score(Fcst)', color='tab:green')
ax1.vlines(forecast.index.min(), 0, pd.concat([df_train['sales'],df_test['sales']]).max()*1.2, linestyle='--', color='tab:red', label='Start of Prediction')

# 각 축의 범례 핸들과 라벨을 가져와서 결합
ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
ax1.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels, loc='upper left')

ax1.set_title(sel_category)
plt.show()

# 모델 적합성 평가(AIC)
print(model_fit.summary())

fig, ax = plt.subplots(figsize=(10,5)) # 신뢰구간 내 자기상관이 존재하므로 적합
sgt.plot_acf(model_fit.resid['sales'],ax=ax)
plt.show()

# 정확도 평가
mae = mean_absolute_error(df_test['sales'], forecast.loc[:df_test.index.max(),'sales'])
print("Mean Absolute Error (MAE): %.3f" % mae)

#6.실데이터 예측
df_diff_train_test = pd.concat([df_diff_train,df_diff_test])
df_train_test = pd.concat([df_train,df_test])

# VAR 모델 구축
model = VAR(df_diff_train_test)
model_fit = model.fit(best_lag)

# 예측 수행
forecast_steps = 13
forecast_interval = model_fit.forecast_interval(df_diff_train_test.values,steps=forecast_steps, alpha=0.05)
forecast = pd.DataFrame(forecast_interval[0], columns=['sales', 'review_score'])
forecast.index = pd.date_range(start=df_diff_train_test.index.max() + timedelta(days=7), periods=forecast_steps, freq='W')
forecast = forecast.rename_axis("date")
forecast['sales'] = forecast['sales'].round(2)

# forecast 역차분
forecast.loc[forecast.index.min(),'sales'] += df_train['sales'].iloc[-1]
forecast['sales'] = forecast['sales'].cumsum()

# 예측 시각화
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(pd.concat([pd.concat([df_train['sales'],df_test['sales']]),forecast[forecast.index==forecast.index.min()]['sales']]), label='Sales(Actual)', color='tab:blue')
ax1.plot(forecast['sales'], label='Sales(Fcst)', color='tab:orange')
ax2 = ax1.twinx()
ax2.plot(pd.concat([pd.concat([df_train['review_score'],df_test['review_score']]),forecast[forecast.index==forecast.index.min()]['review_score']]), label='Review Score(Actual)', color='tab:green', alpha=0.3)
ax2.plot(forecast['review_score'], label='Review Score(Fcst)', color='tab:green')
ax1.vlines(forecast.index.min(), 0, pd.concat([df_train['sales'],df_test['sales']]).max()*1.2, linestyle='--', color='tab:red', label='Start of Prediction')

# 각 축의 범례 핸들과 라벨을 가져와서 결합
ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
ax1.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels, loc='upper left')

ax1.set_title(sel_category)

# plt.savefig('Olist/result/var/%s.png' %sel_category)
plt.show()