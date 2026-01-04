# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun'] # 中文使用宋体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题
from matplotlib.ticker import MaxNLocator

# import seaborn as sns
from patsy import dmatrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
import warnings
warnings.filterwarnings('ignore')

import pickle


# %%
class LassoRegression:
    """
    扩展版本：添加特征效应分析和可视化
    """
    
    def __init__(self, n_knots=9, train_years=10, test_years=1, min_samples=1000):
        """
        参数:
        n_knots: 样条节点数
        train_years: 训练年数
        test_years: 测试年数
        min_samples: 最小样本量
        """
        self.n_knots = n_knots
        self.train_years = train_years
        self.test_years = test_years
        self.min_samples = min_samples
        self.spline_info = {}
        
    def rank_transform(self, df, factor_cols):
        """
        排序变换
        """
        print("进行排序变换...")
        
        df_transformed = df.copy()
        
        for factor in factor_cols:
            rank_col = f'{factor}_rank'
            df_transformed[rank_col] = df_transformed.groupby('year_month')[factor].transform(
                lambda x: x.rank(pct=True, method='average')
            )
            df_transformed[rank_col] = df_transformed[rank_col].clip(0.001, 0.999)
        
        return df_transformed
    
    def create_splines_for_data(self, data, factor_cols):
        """
        为数据集创建样条特征
        """
        spline_features = []
        feature_names = []
        
        for factor in factor_cols:
            rank_col = f'{factor}_rank'
            
            if rank_col not in data.columns:
                continue
            
            try:
                # 样条基数量 = knots + 2
                basis_df = self.n_knots + 2
                
                spline_basis = dmatrix(
                    f"cr(x, df={basis_df}, constraints='center')", 
                    {"x": data[rank_col]}, 
                    return_type='dataframe'
                )
                
                # 列命名
                new_names = [f"{factor}_spline_{i}" for i in range(spline_basis.shape[1])]
                spline_basis.columns = new_names
                
                spline_features.append(spline_basis)
                feature_names.extend(new_names)
                
                # 保存样条信息
                if factor not in self.spline_info:
                    self.spline_info[factor] = {
                        'rank_col': rank_col,
                        'spline_cols': new_names,
                        'formula': f"cr(x, df={basis_df}, constraints='center')",
                        'spline_type': 'cubic_spline'
                    }
                
            except Exception as e:
                print(f"  {factor}: 样条创建失败，使用线性项 - {e}")
                # 使用原始值作为备选
                linear_feature = pd.DataFrame({f"{factor}_linear": data[rank_col]})
                spline_features.append(linear_feature)
                feature_names.append(f"{factor}_linear")
                
                if factor not in self.spline_info:
                    self.spline_info[factor] = {
                        'rank_col': rank_col,
                        'spline_cols': [f"{factor}_linear"],
                        'formula': 'linear',
                        'spline_type': 'linear'
                    }
        
        if spline_features:
            X = pd.concat(spline_features, axis=1)
            return X, feature_names
        else:
            return pd.DataFrame(), []
    
    def rolling_window_estimation(self, df, factor_cols, target_col='excess_return'):
        """
        滚动窗口估计
        """
        print(f"\n开始滚动窗口估计...")
        print(f"训练窗口: {self.train_years}年, 测试窗口: {self.test_years}年")
        
        # 1. 准备数据
        df_prepared = self.rank_transform(df, factor_cols)
        df_prepared = df_prepared.sort_values(['code', 'year_month'])
        
        # 2. 获取时间范围
        unique_months = sorted(df_prepared['year_month'].unique())
        total_months = len(unique_months)
        
        train_months_count = self.train_years * 12
        test_months_count = self.test_years * 12
        step_months = self.test_years * 12
        
        print(f"总月份数: {total_months}")
        print(f"训练月份: {train_months_count}, 测试月份: {test_months_count}")
        
        # 3. 结果存储
        window_results = []
        all_selection_counts = {}
        models_info = []
        
        # 4. 主循环
        window_id = 0
        
        for start_idx in range(0, total_months - train_months_count - test_months_count, step_months):
            window_id += 1
            
            # 训练期
            train_start_idx = start_idx
            train_end_idx = start_idx + train_months_count - 1
            train_months = unique_months[train_start_idx:train_end_idx + 1]
            
            # 测试期
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + test_months_count - 1
            
            if test_end_idx >= total_months:
                break
                
            test_months = unique_months[test_start_idx:test_end_idx + 1]
            
            # 获取数据
            train_data = df_prepared[df_prepared['year_month'].isin(train_months)]
            test_data = df_prepared[df_prepared['year_month'].isin(test_months)]
            
            # 检查数据量
            if len(train_data) < self.min_samples or len(test_data) < self.min_samples/10:
                print(f"窗口 {window_id}: 数据不足，跳过")
                continue
            
            print(f"\n窗口 {window_id}:")
            print(f"  训练期: {train_months[0].strftime('%Y-%m')} 到 {train_months[-1].strftime('%Y-%m')} ({len(train_data):,} 条)")
            print(f"  测试期: {test_months[0].strftime('%Y-%m')} 到 {test_months[-1].strftime('%Y-%m')} ({len(test_data):,} 条)")
            
            # 清空样条信息
            self.spline_info = {}
            
            # 5. 训练数据样条
            try:
                X_train, feature_names = self.create_splines_for_data(train_data, factor_cols)
                y_train = train_data[target_col].values
                
                if len(feature_names) == 0:
                    print(f"  错误: 没有创建任何特征")
                    continue
                    
            except Exception as e:
                print(f"  训练数据样条创建失败: {e}")
                continue
            
            # 6. 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 7. LASSO训练
            try:
                from sklearn.model_selection import TimeSeriesSplit
                
                tscv = TimeSeriesSplit(n_splits=3)
                lasso_cv = LassoCV(
                    cv=tscv,
                    max_iter=5000,
                    tol=1e-4,
                    random_state=42,
                    n_jobs=-1,
                    selection='random'
                )
                
                lasso_cv.fit(X_train_scaled, y_train)
                
                # 最终模型
                final_lasso = Lasso(
                    alpha=lasso_cv.alpha_,
                    max_iter=5000,
                    random_state=42,
                    selection='random'
                )
                final_lasso.fit(X_train_scaled, y_train)
                
            except Exception as e:
                print(f"  LASSO训练失败: {e}")
                try:
                    final_lasso = Lasso(alpha=0.001, max_iter=5000)
                    final_lasso.fit(X_train_scaled, y_train)
                except:
                    continue
            
            # 8. 特征选择统计
            selected_mask = np.abs(final_lasso.coef_) > 1e-6
            selected_features = np.array(feature_names)[selected_mask]
            selected_coefs = final_lasso.coef_[selected_mask]
            
            # 存储每个特征的详细信息
            feature_details = {}
            for feat, coef in zip(feature_names, final_lasso.coef_):
                feature_details[feat] = {
                    'coefficient': coef,
                    'selected': abs(coef) > 1e-6
                }
            
            # 更新选择计数
            for feat, coef in zip(selected_features, selected_coefs):
                if feat not in all_selection_counts:
                    all_selection_counts[feat] = {'count': 0, 'total_coef': 0}
                all_selection_counts[feat]['count'] += 1
                all_selection_counts[feat]['total_coef'] += abs(coef)
            
            # 9. 测试数据预测
            try:
                X_test_list = []
                test_feature_names = []
                
                for factor in factor_cols:
                    if factor in self.spline_info:
                        info = self.spline_info[factor]
                        rank_col = info['rank_col']
                        
                        if rank_col not in test_data.columns:
                            test_data[rank_col] = test_data[factor].rank(pct=True)
                            test_data[rank_col] = test_data[rank_col].clip(0.001, 0.999)
                        
                        if info['formula'] == 'linear':
                            test_feat = pd.DataFrame({info['spline_cols'][0]: test_data[rank_col]})
                            X_test_list.append(test_feat)
                            test_feature_names.append(info['spline_cols'][0])
                        else:
                            try:
                                test_basis = dmatrix(
                                    info['formula'],
                                    {"x": test_data[rank_col]}, 
                                    return_type='dataframe'
                                )
                                
                                if test_basis.shape[1] == len(info['spline_cols']):
                                    test_basis.columns = info['spline_cols']
                                    X_test_list.append(test_basis)
                                    test_feature_names.extend(info['spline_cols'])
                                else:
                                    placeholder = pd.DataFrame(
                                        0, 
                                        index=range(len(test_data)), 
                                        columns=info['spline_cols']
                                    )
                                    X_test_list.append(placeholder)
                                    test_feature_names.extend(info['spline_cols'])
                                    
                            except Exception as e:
                                placeholder = pd.DataFrame(
                                    0, 
                                    index=range(len(test_data)), 
                                    columns=info['spline_cols']
                                )
                                X_test_list.append(placeholder)
                                test_feature_names.extend(info['spline_cols'])
                
                if X_test_list:
                    X_test = pd.concat(X_test_list, axis=1)
                    X_test = X_test.reindex(columns=feature_names, fill_value=0)
                    
                    X_test_scaled = scaler.transform(X_test)
                    y_test = test_data[target_col].values
                    
                    y_pred = final_lasso.predict(X_test_scaled)
                    
                    from sklearn.metrics import r2_score, mean_squared_error
                    
                    test_r2 = r2_score(y_test, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    monthly_r2 = []
                    for month in test_months:
                        month_mask = test_data['year_month'] == month
                        if month_mask.sum() > 10:
                            month_y_test = y_test[month_mask]
                            month_y_pred = y_pred[month_mask]
                            if len(np.unique(month_y_test)) > 1:
                                month_r2 = r2_score(month_y_test, month_y_pred)
                                monthly_r2.append(month_r2)
                    
                    avg_monthly_r2 = np.mean(monthly_r2) if monthly_r2 else 0
                    
                else:
                    test_r2 = 0
                    test_rmse = 0
                    avg_monthly_r2 = 0
                    
            except Exception as e:
                print(f"  测试预测失败: {e}")
                test_r2 = 0
                test_rmse = 0
                avg_monthly_r2 = 0
            
            # 10. 存储结果
            result = {
                'window_id': window_id,
                'train_start': train_months[0],
                'train_end': train_months[-1],
                'test_start': test_months[0],
                'test_end': test_months[-1],
                'n_train_samples': len(train_data),
                'n_test_samples': len(test_data),
                'n_total_features': len(feature_names),
                'n_selected_features': len(selected_features),
                'train_r2': final_lasso.score(X_train_scaled, y_train),
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'avg_monthly_r2': avg_monthly_r2,
                'alpha': final_lasso.alpha,
                'selected_features': list(selected_features)
            }
            
            window_results.append(result)
            
            # 存储模型信息
            models_info.append({
                'window_id': window_id,
                'model': final_lasso,
                'scaler': scaler,
                'feature_names': feature_names,
                'feature_details': feature_details,
                'spline_info': self.spline_info.copy(),  # 保存样条信息
                'train_months': train_months,
                'test_months': test_months,
                'train_data': train_data.copy(),
                'test_data': test_data.copy()
            })
            
            print(f"  结果: 选中 {len(selected_features)} 特征, "
                  f"训练R_square={result['train_r2']:.4f}, "
                  f"测试R_square={result['test_r2']:.4f}")
        
        # 11. 转换为DataFrame
        results_df = pd.DataFrame(window_results)
        
        # 12. 整理特征选择计数
        selection_summary = {}
        for feat, counts in all_selection_counts.items():
            selection_summary[feat] = {
                'selection_count': counts['count'],
                'avg_abs_coef': counts['total_coef'] / counts['count'] if counts['count'] > 0 else 0
            }
        
        return results_df, selection_summary, models_info
    
    def analyze_and_visualize(self, results_df, selection_summary):
        """
        分析和可视化结果
        """
        print(f"\n{'='*60}")
        print("结果分析")
        print(f"{'='*60}")
        
        if results_df.empty:
            print("没有有效结果")
            return
        
        # 1. 基本统计
        print(f"总窗口数: {len(results_df)}")
        print(f"平均训练样本: {results_df['n_train_samples'].mean():.0f}")
        print(f"平均测试样本: {results_df['n_test_samples'].mean():.0f}")
        print(f"平均选中特征数: {results_df['n_selected_features'].mean():.1f}")
        print(f"平均训练R_square: {results_df['train_r2'].mean():.4f}")
        print(f"平均测试R_square: {results_df['test_r2'].mean():.4f}")
        print(f"平均测试月度R_square: {results_df['avg_monthly_r2'].mean():.4f}")
        
        # 2. 特征选择分析
        if selection_summary:
            selection_df = pd.DataFrame(selection_summary).T
            selection_df = selection_df.sort_values('selection_count', ascending=False)
            
            def extract_factor_name(feature_name):
                if '_spline_' in feature_name:
                    return feature_name.split('_spline_')[0]
                elif '_linear' in feature_name:
                    return feature_name.replace('_linear', '')
                else:
                    return feature_name
            
            selection_df['factor'] = selection_df.index.map(extract_factor_name)
            
            factor_summary = selection_df.groupby('factor').agg({
                'selection_count': 'sum',
                'avg_abs_coef': 'mean'
            }).sort_values('selection_count', ascending=False)
            
            print(f"\n特征选择摘要:")
            print(f"总特征数: {len(selection_df)}")
            print(f"总选择次数: {selection_df['selection_count'].sum()}")
            
            print(f"\n最常被选中的因子:")
            print(factor_summary.head(10).to_string())
            
            # 3. 可视化
            try:
                # plt.style.use('seaborn-v0_8-darkgrid')
                fig, axes = plt.subplots(2, 2, figsize=(10, 6))
                
                # 图1: 测试R_square随时间变化
                if 'test_start' in results_df.columns:
                    axes[0, 0].plot(range(len(results_df)), results_df['test_r2'], 'b-o', linewidth=2)
                    axes[0, 0].set_xlabel('窗口编号', fontsize=1)
                    axes[0, 0].set_ylabel('测试$R^2$', fontsize=13.5)
                    axes[0, 0].set_title('测试$R^2$随窗口变化', fontsize=15)
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
                    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    axes[0, 0].set_ylim(bottom=0)
                
                # 图2: 特征选择数
                axes[0, 1].plot(range(len(results_df)), results_df['n_selected_features'], 'g-s', linewidth=2)
                axes[0, 1].set_xlabel('窗口编号', fontsize=1)
                axes[0, 1].set_ylabel('选中特征数', fontsize=1)
                axes[0, 1].set_title('特征选择数变化', fontsize=15)
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
                axes[0, 1].set_ylim(bottom=0)
                
                # 图3: 最常被选中的因子
                top_n = min(10, len(factor_summary))
                top_factors = factor_summary.head(top_n)
                
                y_pos = range(len(top_factors))
                axes[1, 0].barh(y_pos, top_factors['selection_count'].values, color='steelblue')
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels(top_factors.index, fontsize=1)
                axes[1, 0].set_xlabel('被选中次数', fontsize=1)
                axes[1, 0].set_title(f'特征的非线性分布', fontsize=15)
                axes[1, 0].invert_yaxis()
                
                # 图4: 训练vs测试R^2
                axes[1, 1].scatter(results_df['train_r2'], results_df['test_r2'], alpha=0.6, s=50)
                axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
                axes[1, 1].set_xlabel('训练$R^2$', fontsize=1)
                axes[1, 1].set_ylabel('测试$R^2$', fontsize=1)
                axes[1, 1].set_title('训练vs测试$R^2$', fontsize=15)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('normal_lasso_result/lasso_analysis_results.pdf', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\n可视化结果已保存为 'normal_lasso_result/lasso_analysis_results.pdf'")
                
            except Exception as e:
                print(f"可视化失败: {e}")
        
        return factor_summary
    
    def reconstruct_marginal_effect(self, feature_name, model_info, n_points=100, fixed_at_median=True):
        """
        重构一个特征的边际效应函数（条件关系）
        
        参数：
        feature_name: 特征名
        model_info: 单个窗口的模型信息
        n_points: 网格点数
        fixed_at_median: 是否将其他特征固定在中间值
        
        返回：
        x_grid: 特征值网格（标准化后的排名）
        y_effect: 边际效应值
        """
        try:
            # 检查该特征是否在模型中
            spline_info = model_info['spline_info'].get(feature_name)
            if not spline_info:
                print(f"特征 {feature_name} 未在样条信息中找到")
                return None, None
            
            # 创建特征值网格
            x_grid = np.linspace(0.01, 0.99, n_points)
            
            # 创建该特征的基函数
            if spline_info['spline_type'] == 'linear':
                # 线性特征
                basis_values = x_grid.reshape(-1, 1)
                basis_names = spline_info['spline_cols']
            else:
                # 样条特征
                basis_df = pd.DataFrame({'x': x_grid})
                try:
                    basis_matrix = dmatrix(
                        spline_info['formula'],
                        {"x": basis_df['x']}, 
                        return_type='dataframe'
                    )
                    if basis_matrix.shape[1] == len(spline_info['spline_cols']):
                        basis_matrix.columns = spline_info['spline_cols']
                        basis_values = basis_matrix.values
                        basis_names = spline_info['spline_cols']
                    else:
                        print(f"基函数维度不匹配: {feature_name}")
                        return None, None
                except Exception as e:
                    print(f"创建基函数失败: {feature_name}, {e}")
                    return None, None
            
            # 获取系数
            coefficients = []
            for basis_name in basis_names:
                if basis_name in model_info['feature_details']:
                    coefficients.append(model_info['feature_details'][basis_name]['coefficient'])
                else:
                    coefficients.append(0)
            
            coefficients = np.array(coefficients)
            
            # 计算边际效应
            if basis_values.shape[1] == len(coefficients):
                y_effect = basis_values @ coefficients
            else:
                # 维度匹配
                min_dim = min(basis_values.shape[1], len(coefficients))
                y_effect = basis_values[:, :min_dim] @ coefficients[:min_dim]
            
            return x_grid, y_effect
            
        except Exception as e:
            print(f"重构边际效应失败: {feature_name}, {e}")
            return None, None
    
    def compute_unconditional_relationship(self, feature_name, data, target_col='excess_return', n_bins=20):
        """
        计算无条件关系（分组平均）
        
        参数：
        feature_name: 特征名
        data: 数据
        target_col: 目标变量
        n_bins: 分组数
        
        返回：
        bin_centers: 分组中心
        bin_means: 分组平均收益
        bin_stds: 分组标准差
        """
        try:
            # 确保有排名列
            rank_col = f'{feature_name}_rank'
            if rank_col not in data.columns:
                # 创建排名
                data[rank_col] = data[feature_name].rank(pct=True)
                data[rank_col] = data[rank_col].clip(0.001, 0.999)
            
            # 分组
            data['bin'] = pd.cut(data[rank_col], bins=n_bins, labels=False)
            
            # 计算分组统计
            bin_stats = data.groupby('bin').agg({
                rank_col: 'mean',
                target_col: ['mean', 'std', 'count']
            })
            
            bin_stats.columns = ['bin_center', 'return_mean', 'return_std', 'count']
            
            # 过滤样本量太小的组
            bin_stats = bin_stats[bin_stats['count'] > 10]
            
            return bin_stats['bin_center'].values, bin_stats['return_mean'].values, bin_stats['return_std'].values
            
        except Exception as e:
            print(f"计算无条件关系失败: {feature_name}, {e}")
            return None, None, None
    
    def plot_characteristic_effect(self, feature_name, models_info, data, 
                                   target_col='excess_return', n_points=100, n_bins=20,
                                   recent_window_only=True, save_path=None):
        """
        绘制单个特征的非参数关系图（类似论文图7-11）
        
        参数：
        feature_name: 要分析的特征名
        models_info: 存储的模型信息列表
        data: 原始数据（用于计算无条件关系）
        target_col: 目标变量
        n_points: 条件关系网格点数
        n_bins: 无条件关系分组数
        recent_window_only: 是否只使用最近一个窗口
        save_path: 保存路径
        """
        print(f"\n{'='*60}")
        print(f"分析特征: {feature_name}")
        print(f"{'='*60}")
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
        # plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 左图：无条件关系
        print("计算无条件关系...")
        bin_centers, bin_means, bin_stds = self.compute_unconditional_relationship(
            feature_name, data, target_col, n_bins
        )
        
        if bin_centers is not None:
            # 绘制散点（分组平均）
            axes[0].scatter(bin_centers, bin_means, s=50, alpha=0.7, color='steelblue', 
                          edgecolors='white', linewidth=1, label='分组平均')
            
            # 绘制平滑曲线
            from scipy.interpolate import interp1d
            if len(bin_centers) > 3:
                try:
                    # 排序
                    sort_idx = np.argsort(bin_centers)
                    sorted_x = bin_centers[sort_idx]
                    sorted_y = bin_means[sort_idx]
                    
                    # 插值
                    f = interp1d(sorted_x, sorted_y, kind='cubic', fill_value='extrapolate')
                    x_smooth = np.linspace(0, 1, 200)
                    y_smooth = f(x_smooth)
                    
                    axes[0].plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8, label='平滑曲线')
                except:
                    # 线性插值作为备选
                    axes[0].plot(bin_centers, bin_means, 'r-', linewidth=2, alpha=0.8, label='连接线')
            
            axes[0].set_xlabel(f'{feature_name} (排名)', fontsize=12)
            axes[0].set_ylabel(f'平均超额收益', fontsize=12)
            axes[0].set_title(f'无条件关系: {feature_name} vs 超额收益', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 2. 右图：条件关系（边际效应）
        print("计算条件关系...")
        
        if recent_window_only:
            # 使用最近一个窗口
            model_info = models_info[-1]
            x_grid, y_effect = self.reconstruct_marginal_effect(feature_name, model_info, n_points)
            
            if x_grid is not None:
                axes[1].plot(x_grid, y_effect, 'b-', linewidth=3, label='边际效应')
                axes[1].fill_between(x_grid, y_effect - 0.1, y_effect + 0.1, alpha=0.2, color='blue')
                
                # 添加窗口信息
                window_id = model_info['window_id']
                train_months = model_info['train_months']
                axes[1].text(0.05, 0.95, f'窗口 {window_id}\n训练期: {train_months[0].strftime("%Y-%m")} 到 {train_months[-1].strftime("%Y-%m")}',
                           transform=axes[1].transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # 使用所有窗口的平均
            all_x_grid = None
            all_y_effects = []
            
            for model_info in models_info:
                x_grid, y_effect = self.reconstruct_marginal_effect(feature_name, model_info, n_points)
                if x_grid is not None and y_effect is not None:
                    all_x_grid = x_grid
                    all_y_effects.append(y_effect)
            
            if all_y_effects and all_x_grid is not None:
                all_y_effects = np.array(all_y_effects)
                y_mean = np.mean(all_y_effects, axis=0)
                y_std = np.std(all_y_effects, axis=0)
                y_se = y_std / np.sqrt(len(all_y_effects))
                
                axes[1].plot(all_x_grid, y_mean, 'b-', linewidth=3, label='平均边际效应')
                axes[1].fill_between(all_x_grid, y_mean - 1.96*y_se, y_mean + 1.96*y_se, 
                                    alpha=0.3, color='blue', label='95% 置信区间')
        
        axes[1].set_xlabel(f'{feature_name} (排名)', fontsize=12)
        axes[1].set_ylabel(f'边际效应 (关于超额收益)', fontsize=12)
        axes[1].set_title(f'条件关系: {feature_name} (控制其他特征)', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")
        
        plt.show()
        
        # 3. 统计分析
        print("\n统计分析:")
        if x_grid is not None and y_effect is not None:
            # 计算单调性
            diffs = np.diff(y_effect)
            increasing = np.sum(diffs > 0)
            decreasing = np.sum(diffs < 0)
            
            print(f"  单调性: 递增部分 = {increasing}/{len(diffs)}, 递减部分 = {decreasing}/{len(diffs)}")
            
            # 计算极值
            max_idx = np.argmax(y_effect)
            min_idx = np.argmin(y_effect)
            print(f"  最大值位置: x={x_grid[max_idx]:.3f}, y={y_effect[max_idx]:.6f}")
            print(f"  最小值位置: x={x_grid[min_idx]:.3f}, y={y_effect[min_idx]:.6f}")
            
            # 计算形状特征
            if y_effect[0] > y_effect[-1]:
                print(f"  总体趋势: 递减 ({(y_effect[0] - y_effect[-1]):.6f})")
            elif y_effect[0] < y_effect[-1]:
                print(f"  总体趋势: 递增 ({(y_effect[-1] - y_effect[0]):.6f})")
            else:
                print(f"  总体趋势: 平坦")
    
    def analyze_nonlinearity(self, models_info, top_n_features=5):
        """
        分析非线性特征
        
        参数：
        models_info: 模型信息列表
        top_n_features: 分析前N个特征
        """
        print(f"\n{'='*60}")
        print("非线性特征分析")
        print(f"{'='*60}")
        
        # 统计每个特征被选中的次数
        feature_selection = {}
        
        for model_info in models_info:
            for feat, details in model_info['feature_details'].items():
                if details['selected']:
                    # 提取因子名
                    if '_spline_' in feat:
                        factor_name = feat.split('_spline_')[0]
                        feature_type = 'spline'
                    elif '_linear' in feat:
                        factor_name = feat.replace('_linear', '')
                        feature_type = 'linear'
                    else:
                        factor_name = feat
                        feature_type = 'other'
                    
                    if factor_name not in feature_selection:
                        feature_selection[factor_name] = {
                            'count': 0,
                            'types': set(),
                            'max_coef': 0,
                            'min_coef': 0
                        }
                    
                    feature_selection[factor_name]['count'] += 1
                    feature_selection[factor_name]['types'].add(feature_type)
                    feature_selection[factor_name]['max_coef'] = max(
                        feature_selection[factor_name]['max_coef'], 
                        abs(details['coefficient'])
                    )
        
        # 转换为DataFrame
        nonlinear_df = pd.DataFrame([
            {
                'factor': factor,
                'selection_count': info['count'],
                'has_spline': 'spline' in info['types'],
                'has_linear': 'linear' in info['types'],
                'max_coef': info['max_coef']
            }
            for factor, info in feature_selection.items()
        ])
        
        if nonlinear_df.empty:
            print("没有找到有效的特征选择信息")
            return
        
        # 排序
        nonlinear_df = nonlinear_df.sort_values('selection_count', ascending=False)
        
        # 识别非线性特征
        nonlinear_df['is_nonlinear'] = nonlinear_df['has_spline']
        
        print(f"总因子数: {len(nonlinear_df)}")
        print(f"非线性因子数: {nonlinear_df['is_nonlinear'].sum()}")
        print(f"线性因子数: {(~nonlinear_df['is_nonlinear']).sum()}")
        
        print(f"\n非线性特征:")
        nonlinear_features = nonlinear_df[nonlinear_df['is_nonlinear']].head(top_n_features)
        for _, row in nonlinear_features.iterrows():
            print(f"  {row['factor']}: 选中{row['selection_count']}次, 最大系数={row['max_coef']:.6f}")
        
        print(f"\n线性特征:")
        linear_features = nonlinear_df[~nonlinear_df['is_nonlinear']].head(top_n_features)
        for _, row in linear_features.iterrows():
            print(f"  {row['factor']}: 选中{row['selection_count']}次, 最大系数={row['max_coef']:.6f}")
        
        # 可视化
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # # 图1: 非线性特征分布
        # nonlinear_counts = [
        #     nonlinear_df['is_nonlinear'].sum(),
        #     (~nonlinear_df['is_nonlinear']).sum()
        # ]
        # labels = ['非线性特征', '线性特征']
        # colors = ['lightcoral', 'lightsteelblue']
        
        # axes[0].pie(nonlinear_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        #            startangle=90)
        # axes[0].set_title('非线性 vs 线性特征分布')
        
        # 特征重要性 vs 非线性
        top_factors = nonlinear_df.head(10)
        x_pos = range(len(top_factors))
        
        colors = ['lightcoral' if nl else 'lightsteelblue' for nl in top_factors['is_nonlinear']]
        ax.barh(x_pos, top_factors['selection_count'], color=colors)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(top_factors['factor'])
        # ax.set_xlabel('被选中次数')
        fig.supxlabel('被选中次数', y=0.08)
        # ax.set_title('特征的非线性分布')
        ax.invert_yaxis()
        
        # 添加图例
        # from matplotlib.patches import Patch
        # legend_elements = [
        #     Patch(facecolor='lightcoral', label='非线性'),
        #     Patch(facecolor='lightsteelblue', label='线性')
        # ]
        # ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('normal_lasso_result/factor_selection.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        return nonlinear_df


# %%
# 选择数据年份
def get_data_by_year(df, start_year=1995, end_year=2024):
    """
    根据 year_month 列提取指定年份的数据
    """
    df_res = df.copy() 
    df_res['year_month'] = pd.to_datetime(df_res['year_month'], format='%Y-%m')
    df_res = df_res[(df_res['year_month'].dt.year >= start_year) & (df_res['year_month'].dt.year <= end_year)]
    return df_res


# 执行全过程
def run_complete_analysis(df, factor_cols, target_col='excess_return'):
    """
    运行完整的分析流程
    """
    # 1. 初始化模型
    lasso_model = LassoRegression(n_knots=9, train_years=10, test_years=1)
    
    # 2. 运行滚动窗口估计
    print("="*60)
    print("阶段1: 运行滚动窗口LASSO估计")
    print("="*60)
    
    results_df, selection_summary, models_info = lasso_model.rolling_window_estimation(
        df, factor_cols, target_col
    )
    
    # 3. 基本分析
    print("\n" + "="*60)
    print("阶段2: 基本结果分析")
    print("="*60)
    
    factor_summary = lasso_model.analyze_and_visualize(results_df, selection_summary)
    
    # 4. 非线性分析
    print("\n" + "="*60)
    print("阶段3: 非线性特征分析")
    print("="*60)
    
    nonlinear_df = lasso_model.analyze_nonlinearity(models_info, top_n_features=8)
    
    # 5. 选择关键特征绘制关系图
    print("\n" + "="*60)
    print("阶段4: 绘制关键特征的关系图")
    print("="*60)
    
    if factor_summary is not None and not factor_summary.empty:
        # 选择最常被选中的前3个特征
        top_features = factor_summary.head(3).index.tolist()
        
        for i, feature_name in enumerate(top_features):
            print(f"\n分析特征 {i+1}/{len(top_features)}: {feature_name}")
            
            save_path = f'normal_lasso_result/feature_effect_{feature_name}.pdf'
            
            lasso_model.plot_characteristic_effect(
                feature_name=feature_name,
                models_info=models_info,
                data=df,
                target_col=target_col,
                n_points=100,
                n_bins=20,
                recent_window_only=True,
                save_path=save_path
            )
    else:
        print("无法获取因子摘要，使用默认特征")
        if factor_cols:
            for feature_name in factor_cols[:2]:  # 使用前两个特征
                print(f"\n分析特征: {feature_name}")
                
                save_path = f'normal_lasso_result/feature_effect_{feature_name}.pdf'
                
                lasso_model.plot_characteristic_effect(
                    feature_name=feature_name,
                    models_info=models_info,
                    data=df,
                    target_col=target_col,
                    n_points=100,
                    n_bins=20,
                    recent_window_only=True,
                    save_path=save_path
                )
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    return lasso_model, results_df, models_info, factor_summary


# %%
if __name__=='__main__':
    df = pd.read_csv("data/preprocessed/aligned_data.csv").iloc[:, 1:]
    factor_cols = [ 
                'MOM_12_2', 'STR_1', 
                'LTR_36_13', 'IdioVol','MAX', 
                'Beta', 'RealizedVol', 'ZeroRatio', 
                'AutoCorr', 'Size', 'BM'
            ]

    # 全流程
    lasso_model, results_df, models_info, factor_summary = run_complete_analysis( 
            df=get_data_by_year(df, 2005, 2024), 
            factor_cols=factor_cols, 
            target_col='excess_return'
    )

    # #  或手动分析特定特征
    # # 分析size因子
    # lasso_model.plot_characteristic_effect(
    #     feature_name='size',
    #     models_info=models_info,
    #     data=df,
    #     target_col='excess_return',
    #     save_path='normal_lasso_result/size_effect.pdf'
    # )

    # 分析非线性特征
    nonlinear_df = lasso_model.analyze_nonlinearity(models_info)

    # 保存结果
    lasso_results = { 
        # 'lasso_model': lasso_model,
        'results_df': results_df,
        'models_info': models_info,
        'factor_summary': factor_summary,
        'nonlinear_df': nonlinear_df
    } 

    # 将结果保存在 pickle 当中
    filename = "normal_lasso_result/lasso_results.pkl"
    with open(filename, 'wb') as f:  # 'wb'表示二进制写入
        pickle.dump(lasso_results, f)
    
    print(f"结果已保存到: {filename}")

