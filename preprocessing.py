# %%
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# %%
# raw 数据的读取

daily_return = pd.read_csv("data/raw/daily_return.csv")
market_return = pd.read_csv("data/raw/market_yield.csv")
risk_free_rate = pd.read_csv("data/raw/risk_free_rate.csv")

# %%

def preprocess_data(daily_return, market_return, risk_free_rate):
    """
    预处理数据: 合并三个数据框, 计算超额收益率, 准备因子计算基础
    """
    # 1. 确保日期格式
    daily_return['date'] = pd.to_datetime(daily_return['date'])
    market_return['date'] = pd.to_datetime(market_return['date'])
    risk_free_rate['date'] = pd.to_datetime(risk_free_rate['date'])
    
    # 2. 计算个股超额收益率: r_i - r_f
    # 首先合并个股收益率和无风险利率
    df = pd.merge(daily_return, risk_free_rate, on='date', how='left')
    df['excess_return'] = df['r_i'] - df['r_f']
    
    # 3. 合并市场收益率
    df = pd.merge(df, market_return, on='date', how='left')
    
    # 4. 计算市场超额收益率
    df['market_excess'] = df['r_m'] - df['r_f']
    
    # 5. 按股票代码分组, 为后续计算做准备
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 6. 添加年月列, 方便按月汇总
    df['year_month'] = df['date'].dt.to_period('M')
    
    return df

# 执行预处理
df_processed = preprocess_data(daily_return, market_return, risk_free_rate)
print(f"预处理后数据形状: {df_processed.shape}")
print(df_processed.head())

# %%
# 因子计算 (月度)

def calculate_monthly_factors(df, min_trading_days=15):
    """
    基于日度数据计算月度因子
    返回: 月度DataFrame, 每行包含股票代码、年月和所有因子值
    """
    factors_list = []
    
    # 按股票代码分组
    num_group = 0 # 处理进度监控
    for code, group in df.groupby('code'):
        group = group.sort_values('date').reset_index(drop=True)

        # 处理进度监控
        num_group += 1
        if num_group % 100 == 0:
            print(f"正在处理股票代码: {code}, 进度为{num_group}/{len(df['code'].unique())}")
        
        # 计算月度数据点（每月最后一个交易日）
        monthly_dates = group.groupby('year_month')['date'].last().reset_index()
        
        for _, row in monthly_dates.iterrows():
            current_date = row['date']
            current_ym = row['year_month']

            # 获取截至当前日期的所有数据
            history = group[group['date'] <= current_date].copy()
            
            if len(history) < 30:  # 至少需要30个交易日
                continue
                
            # 因子1: 动量 (MOM_12_2) - 过去12个月至前2个月的累计收益
            mom_12_2 = calculate_momentum(history, current_date, window_months=12, skip_months=1)
            
            # 因子2: 短期反转 (STR_1) - 上个月的收益率
            str_1 = calculate_reversal(history, current_date, window_months=1)
            
            # 因子3: 长期反转 (LTR_36_13) - 过去36个月至前13个月的累计收益
            ltr_36_13 = calculate_longterm_reversal(history, current_date, window_months=36, skip_months=12)
            
            # 因子4: 特质波动率 (IdioVol) - 过去21个交易日对市场模型回归的残差标准差
            idio_vol = calculate_idio_volatility(history, current_date, window_days=21)
            
            # 因子5: 极端收益 (MAX) - 过去21个交易日的最大日收益率
            max_return = calculate_max_effect(history, current_date, window_days=21)
            
            # 因子6: 市场贝塔 (Beta) - 过去252个交易日的CAPM Beta
            beta = calculate_beta(history, current_date, window_days=252)
            
            # 因子7: 已实现波动率 (RealizedVol) - 过去21个交易日的收益率标准差
            realized_vol = calculate_realized_volatility(history, current_date, window_days=21)
            
            # 因子8: 零收益率比例 (ZeroRatio) - 过去21个交易日收益率为零的天数比例
            zero_ratio = calculate_zero_return_ratio(history, current_date, window_days=21)
            
            # 因子9: 收益率自相关 (AutoCorr) - 过去21个交易日的一阶自相关系数
            autocorr = calculate_autocorrelation(history, current_date, window_days=21)
            
            # 组合因子记录
            factor_record = {
                'code': code,
                'date': current_date,
                'year_month': current_ym,
                'MOM_12_2': mom_12_2,
                'STR_1': str_1,
                'LTR_36_13': ltr_36_13,
                'IdioVol': idio_vol,
                'MAX': max_return,
                'Beta': beta,
                'RealizedVol': realized_vol,
                'ZeroRatio': zero_ratio,
                'AutoCorr': autocorr
            }
            
            factors_list.append(factor_record)
    
    # 转换为DataFrame
    factors_df = pd.DataFrame(factors_list)
    
    # 处理缺失值
    factors_df = factors_df.dropna(subset=['MOM_12_2', 'STR_1', 'Beta'])  # 保留关键因子非缺失的记录
    
    return factors_df

# 各因子的具体计算函数
def calculate_momentum(history, current_date, window_months=12, skip_months=1):
    """
    计算动量因子: 过去window_months个月, 跳过最近skip_months个月的累计收益
    """
    # 计算日期范围
    end_date = current_date - pd.DateOffset(months=skip_months)
    start_date = end_date - pd.DateOffset(months=window_months)
    
    # 获取该时间段的数据
    mask = (history['date'] > start_date) & (history['date'] <= end_date)
    period_data = history.loc[mask, 'excess_return']
    
    if len(period_data) < 10:  # 至少需要10个交易日
        return np.nan
    
    # 计算累计收益
    cum_return = (1 + period_data).prod() - 1
    return cum_return

def calculate_reversal(history, current_date, window_months=1):
    """
    计算短期反转: 最近window_months个月的累计收益
    """
    start_date = current_date - pd.DateOffset(months=window_months)
    
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    period_data = history.loc[mask, 'excess_return']
    
    if len(period_data) < 5:
        return np.nan
    
    cum_return = (1 + period_data).prod() - 1
    return cum_return

def calculate_longterm_reversal(history, current_date, window_months=36, skip_months=12):
    """
    计算长期反转: 过去window_months个月, 跳过最近skip_months个月的累计收益
    """
    end_date = current_date - pd.DateOffset(months=skip_months)
    start_date = end_date - pd.DateOffset(months=window_months)
    
    mask = (history['date'] > start_date) & (history['date'] <= end_date)
    period_data = history.loc[mask, 'excess_return']
    
    if len(period_data) < 60:  # 至少需要60个交易日
        return np.nan
    
    cum_return = (1 + period_data).prod() - 1
    return cum_return

def calculate_idio_volatility(history, current_date, window_days=21):
    """
    计算特质波动率: 过去window_days个交易日对市场模型回归的残差标准差
    """
    # 获取最近window_days个交易日的数据
    start_date = current_date - pd.Timedelta(days=window_days*2)  # 确保有足够数据
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    recent_data = history.loc[mask].copy()
    
    if len(recent_data) < window_days:
        return np.nan
    
    # 取最近window_days个交易日
    recent_data = recent_data.tail(window_days)
    
    # 对市场模型回归: r_i = alpha + beta * r_m + epsilon
    X = recent_data['market_excess'].values.reshape(-1, 1)
    y = recent_data['excess_return'].values
    
    # 简单OLS回归（实际可用statsmodels, 这里用numpy简化）
    try:
        beta = np.cov(y, X.flatten())[0, 1] / np.var(X.flatten())
        residuals = y - beta * X.flatten()
        idio_vol = np.std(residuals)
        return idio_vol
    except:
        return np.nan

def calculate_max_effect(history, current_date, window_days=21):
    """
    计算MAX效应: 过去window_days个交易日的最大日收益率
    """
    start_date = current_date - pd.Timedelta(days=window_days*2)
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    recent_data = history.loc[mask, 'excess_return']
    
    if len(recent_data) < window_days:
        return np.nan
    
    recent_data = recent_data.tail(window_days)
    max_return = recent_data.max()
    return max_return

def calculate_beta(history, current_date, window_days=252):
    """
    计算CAPM Beta: 过去window_days个交易日的回归Beta
    """
    start_date = current_date - pd.Timedelta(days=window_days*1.5)
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    recent_data = history.loc[mask].copy()
    
    if len(recent_data) < window_days:
        return np.nan
    
    recent_data = recent_data.tail(window_days)
    
    X = recent_data['market_excess'].values
    y = recent_data['excess_return'].values
    
    try:
        # Beta = cov(r_i, r_m) / var(r_m)
        beta = np.cov(y, X)[0, 1] / np.var(X)
        return beta
    except:
        return np.nan

def calculate_realized_volatility(history, current_date, window_days=21):
    """
    计算已实现波动率: 过去window_days个交易日的收益率标准差
    """
    start_date = current_date - pd.Timedelta(days=window_days*2)
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    recent_data = history.loc[mask, 'excess_return']
    
    if len(recent_data) < window_days:
        return np.nan
    
    recent_data = recent_data.tail(window_days)
    realized_vol = recent_data.std()
    return realized_vol

def calculate_zero_return_ratio(history, current_date, window_days=21, threshold=0.001):
    """
    计算零收益率比例: 过去window_days个交易日中|收益率|<threshold的天数比例
    """
    start_date = current_date - pd.Timedelta(days=window_days*2)
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    recent_data = history.loc[mask, 'excess_return']
    
    if len(recent_data) < window_days:
        return np.nan
    
    recent_data = recent_data.tail(window_days)
    zero_count = (recent_data.abs() < threshold).sum()
    zero_ratio = zero_count / len(recent_data)
    return zero_ratio

def calculate_autocorrelation(history, current_date, window_days=21):
    """
    计算一阶自相关系数: 过去window_days个交易日收益率的一阶自相关
    """
    start_date = current_date - pd.Timedelta(days=window_days*2)
    mask = (history['date'] > start_date) & (history['date'] <= current_date)
    recent_data = history.loc[mask, 'excess_return']
    
    if len(recent_data) < window_days:
        return np.nan
    
    recent_data = recent_data.tail(window_days).values
    
    if len(recent_data) >= 2:
        # 计算一阶自相关系数
        autocorr = np.corrcoef(recent_data[:-1], recent_data[1:])[0, 1]
        return autocorr if not np.isnan(autocorr) else 0
    else:
        return 0

# %%
# 计算所有因子（这可能需要一些时间，取决于数据量）
print("开始计算因子...")
factors_monthly = calculate_monthly_factors(df_processed)
print(f"因子计算完成！共 {len(factors_monthly)} 条月度记录")

# 查看结果
print(factors_monthly.head())
print(f"\n数据形状: {factors_monthly.shape}")
print(f"\n时间范围: {factors_monthly['date'].min()} 到 {factors_monthly['date'].max()}")
print(f"\n股票数量: {factors_monthly['code'].nunique()}")

# 检查缺失值
print(f"\n各因子缺失值比例:")
print(factors_monthly.isnull().mean().sort_values(ascending=False))

# %%

# factors_monthly.to_csv('data/preprocessed/factors_monthly.csv', index=False)


# %%
# 市值 账面市值比

df_BM_Size = pd.read_csv('data/raw/市值_账面市值比/FI_T10.csv') 
df_BM_Size = df_BM_Size[ df_BM_Size['Source'] == 0 ][ 
    ['Stkcd', 'Accper', 'F100801A', 'F101001A']
]
df_BM_Size.columns = ['code', 'date', 'Size', 'BM']
df_BM_Size['date'] = pd.to_datetime(df_BM_Size['date']) 
df_BM_Size['year_month'] = df_BM_Size['date'].dt.strftime('%Y-%m')
df_BM_Size['year_month'] = pd.to_datetime(df_BM_Size['year_month'], format='%Y-%m')
df_BM_Size.drop(columns='date', inplace=True)

# df_BM_Size.to_csv("data/preprocessed/BM_Size_monthly.csv", index=False)