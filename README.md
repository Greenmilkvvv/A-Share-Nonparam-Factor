# 基于非参数LASSO方法的A股公司特征与收益关系研究

$\text{diyang}$

---

 **注意事项**:  
 > `data/raw` 这个文件夹本应存放原始数据, 但是因为收益率和无风险利率的原始数据是日度的 (后经预处理生成月度数据), 文件太大, 因而仓库里就不放了. 而预处理过程 `preprocessing.py` 依然上传, 但是显然就不容易运行了. 有数据方面问题都欢迎提交 Issues 或者通过邮箱告知. 
 > `data/preprocessed` 文件夹存放的是经过预处理后的数据, 本文主要基于这些数据进行分析

**本文参考**: 
- Joachim Freyberger, Andreas Neuhierl, and Michael Weber. Dissecting characteristics nonparametrically. The Review of Financial Studies, 33(5):2326–2377, 2020.

## 数据处理

### 数据选取

本文对A股多年的日度交易数据进行预处理, 结合同频率的无风险利率数据, 得到**月度**的超额收益率. 

### 特征计算

本文选取 11 个公司特征因子进行研究, 其中**估值**类的 *市值因子 (Size)* 和 *账面市值比 (BM)* 是依托于上市公司披露获取. 在这之后, 基于**日度**频率的交易数据, 我们得以计算另 9 个月度的因子值, 它们涵盖了 **动量与反转**, **风险与波动率**, **流动性**, **市场微观结构**等特征维度.

公司特征因子定义和计算方法如下: 

#### 一、估值类因子
1. **市值（Size）**: 公司流通市值的自然对数, 衡量规模特征
2. **账面市值比（BM）**: 公司账面价值与市场价值的比率, 衡量估值水平

#### 二、动量与反转类因子
3. **动量因子（MOM_12_2）**: 过去12个月累计收益（剔除最近1个月）, 捕捉中期动量效应
4. **短期反转因子（STR_1）**: 最近1个月的累计收益, 捕捉短期价格反转
5. **长期反转因子（LTR_36_13）**: 过去36个月累计收益（剔除最近12个月）, 捕捉长期反转效应

#### 三、风险与波动率类因子
6. **特质波动率（IdioVol）**: 市场模型回归残差的标准差, 衡量公司特有风险
7. **市场贝塔（Beta）**: CAPM模型估计的系统风险系数
8. **已实现波动率（RealizedVol）**: 收益率标准差, 衡量股票总波动性
9. **最大日收益率（MAX）**: 最大单日收益率, 作为"彩票型"股票偏好的代理变量

#### 四、流动性因子
10. **零收益率比例（ZeroRatio）**: 收益率接近零的天数比例, 衡量流动性不足

#### 五、市场微观结构因子
11. **收益率自相关（AutoCorr）**: 收益率的一阶自相关系数, 捕捉序列相关性

## 实证方法 

本文参照 Freyberger et al., (2020) 的方法, 使用非参数LASSO方法来研究A股公司特征与收益的关系. 

### 非参数LASSO方法

在 `lasso.py` 中定义了类 `LassoRegression`, 通过复现两个核心步骤实现**特征选择**与**非线性关系**估计: 

1. **特征的非参数扩展**: 
首先对每个公司特征进行排序变换, 将原始特征值转换为横截面排名百分位, 以消除量纲影响并缓解异常值干扰. 随后, 使用自然三次样条基函数对每个变换后的特征进行扩展: 设置9个节点将特征值域等分为10个区间, 每个特征被扩展为12个样条基函数, 允许模型捕捉特征与收益之间复杂的非线性关系. 

2. **LASSO正则化与特征选择**: 
将所有特征的样条基函数组合形成高维设计矩阵, 通过LASSO（Least Absolute Shrinkage and Selection Operator）正则化同时进行特征选择与系数估计. LASSO通过在损失函数中添加L1范数惩罚项, 将不重要的基函数系数压缩至零, 从而实现自动特征选择. 模型目标函数为: 
$$
\hat{\beta} = \underset{\beta}{\arg\min} \left\{ \frac{1}{2N} \sum_{i=1}^N \left( R_i - \sum_{j=1}^P x_{ij} \beta_j \right)^2 + \alpha \sum_{j=1}^P |\beta_j| \right\}
$$
其中$\alpha$为正则化参数, 通过时间序列交叉验证自动选择. 

### 滚动窗口估计框架

为检验模型的时变稳定性与样本外预测能力, 本文采用滚动窗口估计策略**:** 训练窗口长度为10年（120个月）, 测试窗口为1年（12个月）. 在每个窗口中, 使用训练期数据估计模型参数, 然后在测试期进行样本外预测. 

**说明**: 由于本文数据的总时间长度并不是很大, 且数据因计算方法等限制而存在缺失值, 所以只选取了 3 个窗口进行训练和测试. 

### 边际效应重构与可视化

基于估计的样条系数, 重构每个特征的边际效应函数, 展示在控制其他特征的条件下, 该特征与预期收益之间的非线性关系. 通过对比无条件关系（简单分组平均）与条件关系（边际效应）, 深入揭示特征的真实预测模式. 

该方法的核心优势在于: 既能处理高维特征间的多重共线性问题, 又能灵活捕捉非线性关系, 同时通过正则化避免过拟合, 为理解A股市场横截面收益的决定因素提供了更为精细的分析框架. 

## 程序说明

请首先安装 `Python`, 然后安装依赖包: 

```bash
pip install -r requirements.txt
```

然后执行下面的命令: 

```bash
python lasso.py
```

也可以通过在项目中简单修改代码, 尝试在不同时间窗口下运行, 或者进行不同的特征选择和模型估计.


## 项目结构

```apl
A-Share-Nonparam-Factor "项目仓库"
├─ preprocessing.py "数据预处理和因子计算"
├─ aligning_data.py "数据对齐与合并"
├─ exploratory_analyses.ipynb "简单的探索性分析"
├─ lasso.py "非参数LASSO模型估计"
├─ description_visulization_conclusion.ipynb "描述性统计、可视化与分析结果呈现"
├─ data "数据文件夹"
│  ├─ preprocessed "预处理后的数据"
│  │  ├─ excess_returns_monthly.csv "月度超额收益率"
│  │  ├─ BM_Size_monthly.csv "估值类的 2 个因子"
│  │  ├─ factors_monthly.csv "其他需计算得到的 9 个因子"
│  │  └─ aligned_data.csv "对齐后的数据"
│  └─ raw "原始数据"
│     └─ raw_data_not_provided "空文件夹 原始数据未提供"
└─ normal_lasso_result "非参数LASSO结果与可视化文件夹"
   ├─ lasso_results.pkl "LASSO结果"
   ├─ factor_selection.pdf "特征选择结果"
   ├─ feature_effect_MOM_12_2.pdf "MOM_12_2 特征效应"
   ├─ feature_effect_STR_1.pdf "STR_1 特征效应"
   ├─ feature_effect_ZeroRatio.pdf "ZeroRatio 特征效应"
   ├─ lasso_analysis_results.pdf "LASSO分析结果"
   ├─ time_varying_effect_STR_1.pdf "STR_1 时变效应"
   ├─ time_varying_effect_ZeroRatio.pdf "ZeroRatio 时变效应"
   ├─ 上市公司数量与上证指数.pdf "上市公司数量与上证指数"
   └─ 因子热力图.pdf "因子相关性热力图"

```

## 联系我们

如有任何问题, 或者需要数据, 欢迎通过 (greenmilkvvv@outlook.com) 告知. 

***

谢谢你们能看到这里. 