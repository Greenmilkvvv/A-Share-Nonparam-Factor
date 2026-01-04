# %%
import pandas as pd
import numpy as np
# from datetime import datetime, timedelta

# %%
# 大部分因子
df = pd.read_csv("data/preprocessed/factors_monthly.csv")
df.drop(columns="date", inplace=True)
df['year_month'] = pd.to_datetime(df['year_month'])

# 市值 账面市值比
BM_Size = pd.read_csv("data/preprocessed/BM_Size_monthly.csv")
BM_Size['year_month'] = pd.to_datetime(BM_Size['year_month'])

# %%

def robust_financial_alignment(df, bm_size_df):
    """
    正确的财务数据对齐方法：
    1. 保持df所有记录
    2. 将财务数据对齐到每个月的月末
    3. 向前填充直到新数据出现
    """
    print("="*60)
    print("财务数据对齐（保持原始数据量）")
    print("="*60)
    
    # 1. 备份原始数据
    df_result = df.copy()
    financial = bm_size_df.copy()
    
    print(f"原始df记录数: {len(df_result):,}")
    print(f"财务数据记录数: {len(financial):,}")
    
    # 2. 准备时间信息（确保是datetime格式）
    if not isinstance(df_result['year_month'].iloc[0], pd.Period):
        df_result['date'] = pd.to_datetime(df_result['year_month'].astype(str) + '-01')
        df_result['year_month'] = df_result['date'].dt.to_period('M')
    
    if not isinstance(financial['year_month'].iloc[0], pd.Period):
        financial['date'] = pd.to_datetime(financial['year_month'].astype(str) + '-01')
        financial['year_month'] = financial['date'].dt.to_period('M')
    
    # 3. 方法一：创建完整的月度面板，然后前向填充
    print("\n方法一：创建完整的股票-月度面板...")
    
    # 获取所有唯一的股票和月份
    all_codes = df_result['code'].unique()
    all_months = df_result['year_month'].unique()
    
    print(f"唯一股票数: {len(all_codes)}")
    print(f"唯一月份数: {len(all_months)}")
    
    # 为每个股票创建完整的月度序列
    full_panel_list = []
    for code in all_codes:
        # 这个股票的财务数据
        stock_fin = financial[financial['code'] == code].sort_values('year_month')
        
        if len(stock_fin) == 0:
            # 如果这个股票完全没有财务数据，跳过
            continue
            
        # 创建这个股票的所有月份
        for month in all_months:
            full_panel_list.append({
                'code': code,
                'year_month': month
            })
    
    full_panel = pd.DataFrame(full_panel_list)
    
    # 4. 合并财务数据到完整面板
    financial = financial.rename(columns={'Size': 'Size_raw', 'BM': 'BM_raw'})
    full_panel = pd.merge(
        full_panel,
        financial[['code', 'year_month', 'Size_raw', 'BM_raw']],
        on=['code', 'year_month'],
        how='left'
    )
    
    # 5. 按股票向前填充财务数据
    print("按股票向前填充财务数据...")
    
    full_panel = full_panel.sort_values(['code', 'year_month'])
    full_panel['Size_filled'] = full_panel.groupby('code')['Size_raw'].ffill()
    full_panel['BM_filled'] = full_panel.groupby('code')['BM_raw'].ffill()
    
    # 6. 合并回原始df（保持所有原始记录）
    print(f"完整面板记录数: {len(full_panel):,}")
    
    # 只取我们需要填充的列
    fill_data = full_panel[['code', 'year_month', 'Size_filled', 'BM_filled']].copy()
    
    # 合并到原始df
    df_result = pd.merge(
        df_result,
        fill_data,
        on=['code', 'year_month'],
        how='left'  # 保持df_result的所有记录
    )
    
    # 7. 检查结果
    print(f"\n合并后df记录数: {len(df_result):,}")
    print(f"Size_filled 缺失比例: {df_result['Size_filled'].isna().mean():.2%}")
    print(f"BM_filled 缺失比例: {df_result['BM_filled'].isna().mean():.2%}")
    
    return df_result

def alternative_simple_method(df, bm_size_df):
    """
    更简单直接的方法：在合并前先扩展财务数据
    """
    print("\n" + "="*60)
    print("方法二：简单直接的方法")
    print("="*60)
    
    df_result = df.copy()
    financial = bm_size_df.copy()
    
    # 1. 转换时间格式
    for df_temp in [df_result, financial]:
        if not isinstance(df_temp['year_month'].iloc[0], pd.Period):
            df_temp['year_month'] = pd.to_datetime(df_temp['year_month']).dt.to_period('M')
    
    # 2. 对每个股票单独处理
    all_codes = df_result['code'].unique()
    
    size_list, bm_list = [], []
    
    for i, code in enumerate(all_codes):
        if i % 1000 == 0:
            print(f"处理第 {i}/{len(all_codes)} 个股票...")
        
        # 这个股票在df中的所有月份
        df_months = df_result[df_result['code'] == code]['year_month'].unique()
        
        # 这个股票的财务数据
        fin_data = financial[financial['code'] == code].sort_values('year_month')
        
        if len(fin_data) == 0:
            # 没有财务数据，全部填充NaN
            for month in df_months:
                size_list.append({'code': code, 'year_month': month, 'Size': np.nan})
                bm_list.append({'code': code, 'year_month': month, 'BM': np.nan})
            continue
        
        # 创建财务数据的完整时间序列
        fin_dict = {}
        last_size, last_bm = None, None
        
        # 按时间顺序处理所有月份
        all_months_sorted = sorted(df_months)
        fin_months = sorted(fin_data['year_month'].unique())
        fin_idx = 0
        
        for month in all_months_sorted:
            # 如果当前月份有财务数据，更新最新值
            if fin_idx < len(fin_months) and month >= fin_months[fin_idx]:
                # 找到这个月份对应的财务数据
                current_fin = fin_data[fin_data['year_month'] == fin_months[fin_idx]]
                if len(current_fin) > 0:
                    last_size = current_fin['Size'].iloc[0]
                    last_bm = current_fin['BM'].iloc[0]
                # 移动到下一个财务月份
                while fin_idx < len(fin_months) and month >= fin_months[fin_idx]:
                    fin_idx += 1
            
            # 记录当前月份的值
            size_list.append({'code': code, 'year_month': month, 'Size': last_size})
            bm_list.append({'code': code, 'year_month': month, 'BM': last_bm})
    
    # 3. 创建填充后的财务数据
    size_df = pd.DataFrame(size_list)
    bm_df = pd.DataFrame(bm_list)
    
    # 4. 合并到原始df
    df_result = pd.merge(df_result, size_df, on=['code', 'year_month'], how='left')
    df_result = pd.merge(df_result, bm_df, on=['code', 'year_month'], how='left')
    
    print(f"\n处理后记录数: {len(df_result):,}")
    print(f"Size缺失比例: {df_result['Size'].isna().mean():.2%}")
    print(f"BM缺失比例: {df_result['BM'].isna().mean():.2%}")
    
    return df_result

def quick_debug_merge(df, bm_size_df):
    """
    快速调试：查看合并过程发生了什么
    """
    print("\n" + "="*60)
    print("调试：为什么数据会减少？")
    print("="*60)
    
    # 查看几个示例股票
    sample_codes = df['code'].drop_duplicates().head(3).tolist()
    
    for code in sample_codes:
        print(f"\n股票 {code}:")
        
        # 这个股票在df中的所有月份
        df_months = df[df['code'] == code]['year_month'].nunique()
        print(f"  - 在df中有 {df_months} 个不同月份")
        
        # 这个股票在财务数据中的所有月份
        fin_months = bm_size_df[bm_size_df['code'] == code]['year_month'].nunique()
        print(f"  - 在财务数据中有 {fin_months} 个不同月份")
        
        # 具体月份
        if fin_months > 0:
            fin_dates = bm_size_df[bm_size_df['code'] == code]['year_month'].unique()[:5]
            print(f"  - 财务数据月份示例: {fin_dates}")
    
    # 查看合并发生了什么
    print("\n测试合并逻辑:")
    test_df = df[df['code'].isin(sample_codes)].copy()
    test_fin = bm_size_df[bm_size_df['code'].isin(sample_codes)].copy()
    
    # 直接合并
    merged = pd.merge(test_df, test_fin, on=['code', 'year_month'], how='left')
    print(f"测试合并前: {len(test_df)} 行")
    print(f"测试合并后: {len(merged)} 行")
    print(f"Size有数据的行数: {merged['Size'].notna().sum()}")

# 主执行函数
def main():
    # 先调试
    quick_debug_merge(df, BM_Size)
    
    # 使用方法一
    print("\n" + "="*60)
    print("开始正式数据对齐...")
    print("="*60)
    
    result = robust_financial_alignment(df, BM_Size)
    
    # 保存结果
    # result.to_csv('aligned_data_full.csv', index=False)
    # print(f"\n 数据保存完成！记录数: {len(result):,}")
    
    # 显示样本
    print("\n样本数据-前5行:")
    print(result.head())
    
    # 检查是否有数据减少
    if len(result) < len(df):
        print(f"\n 警告：数据从 {len(df):,} 减少到 {len(result):,}")
        print("可能原因：某些股票在财务数据中没有记录")
    else:
        print(f"\n 成功保持所有 {len(result):,} 条记录")
    
    return result

# 执行
if __name__ == "__main__":
    aligned_factors_data = main()


# %%

aligned_factors_data.rename( 
    { 
    'Size_filled': 'Size',
    'BM_filled': 'BM',
    }, 
    inplace=True
)

aligned_factors_data.drop(columns='date', inplace=True)

aligned_factors_data


# %%
# 接下来处理 月度超额收益率 
# from datetime import datetime

daily_return = pd.read_csv('data/raw/daily_return.csv')
risk_free_rate = pd.read_csv('data/raw/risk_free_rate.csv')

def calculate_monthly_excess_returns(daily_return, risk_free_rate):
    """
    计算月度超额收益率
    
    参数:
    daily_return: DataFrame, 包含 ['date', 'code', 'r_i'] - 日度个股收益率
    risk_free_rate: DataFrame, 包含 ['date', 'r_f'] - 日度无风险利率
    
    返回:
    monthly_excess: DataFrame, 包含 ['code', 'year_month', 'excess_return']
    """
    
    print("开始计算月度超额收益率...")
    
    # 1. 确保日期格式正确
    daily_return['date'] = pd.to_datetime(daily_return['date'])
    risk_free_rate['date'] = pd.to_datetime(risk_free_rate['date'])
    
    # 2. 合并无风险利率
    print("合并无风险利率数据...")
    df_merged = pd.merge(daily_return, risk_free_rate, on='date', how='left')
    
    # 3. 计算日度超额收益率
    print("计算日度超额收益率...")
    df_merged['daily_excess'] = df_merged['r_i'] - df_merged['r_f']
    
    # 4. 添加年月标识
    df_merged['year_month'] = df_merged['date'].dt.to_period('M')
    
    # 5. 按股票和月份计算月度超额收益率
    print("按股票和月份聚合计算月度超额收益率...")
    
    def calculate_monthly_return(group):
        """计算单个股票的单月超额收益率"""
        # 确保按日期排序
        group = group.sort_values('date')
        
        # 月度超额收益率 = ∏(1 + 日超额收益率) - 1
        monthly_excess = (1 + group['daily_excess']).prod() - 1
        
        return pd.Series({
            'excess_return': monthly_excess,
            'trading_days': len(group),
            'first_date': group['date'].iloc[0],
            'last_date': group['date'].iloc[-1]
        })
    
    # 分组计算
    monthly_excess = (
        df_merged.groupby(['code', 'year_month'])
        .apply(calculate_monthly_return)
        .reset_index()
    )
    
    # 6. 数据质量检查
    # print(f"\n数据质量检查:")
    # print(f"总记录数: {len(monthly_excess):,}")
    # print(f"股票数量: {monthly_excess['code'].nunique()}")
    # print(f"月份范围: {monthly_excess['year_month'].min()} 到 {monthly_excess['year_month'].max()}")
    # print(f"交易天数统计:")
    # print(f"  - 平均每月交易天数: {monthly_excess['trading_days'].mean():.1f}")
    # print(f"  - 最小交易天数: {monthly_excess['trading_days'].min()}")
    # print(f"  - 最大交易天数: {monthly_excess['trading_days'].max()}")
    
    # # 7. 过滤交易天数过少的月份（可选，但推荐）
    # # A股每月通常有20-23个交易日，设置一个合理阈值
    # min_trading_days = 10
    # initial_count = len(monthly_excess)
    # monthly_excess = monthly_excess[monthly_excess['trading_days'] >= min_trading_days]
    
    # print(f"\n过滤后记录数变化: {initial_count:,} → {len(monthly_excess):,}")
    # print(f"过滤掉 {initial_count - len(monthly_excess):,} 条记录（交易天数 < {min_trading_days}）")
    
    # 8. 选择需要的列
    result = monthly_excess[['code', 'year_month', 'excess_return']].copy()
    
    # 9. 示例输出
    # print(f"\n前5条记录示例:")
    # print(result.head())
    
    # print(f"\n月度超额收益率统计:")
    # print(f"均值: {result['excess_return'].mean():.4f}")
    # print(f"标准差: {result['excess_return'].std():.4f}")
    # print(f"最小值: {result['excess_return'].min():.4f}")
    # print(f"最大值: {result['excess_return'].max():.4f}")
    
    return result

# 使用函数
monthly_excess_returns = calculate_monthly_excess_returns(daily_return, risk_free_rate)



# %%

# monthly_excess_returns.to_csv('data/preprocessed/excess_returns_monthly.csv', index=False)

# %%

# monthly_excess_returns


# %%

df_new = aligned_factors_data.merge(monthly_excess_returns, on=['code', 'year_month'], how='left')

# %%
df_new = df_new.rename( 
    columns={ 
        'Size_filled': 'Size',
        'BM_filled': 'BM',
    }, 
    copy=False
)

df_new = df_new.dropna() 
# df_new.to_csv("data/preprocessed/aligned_data.csv", index=False)
