#############all feature###############

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
data = pd.read_csv('C:/Users/yandcuo/Box/Dr/project/process/data/SunEnergy1_DataSet/2024/03/Total_Inverter_Results_Analysis4.csv')

# 标准化函数，处理 NaN 值的情况
def standardize(df, cols, mean_std_dict=None):
    if mean_std_dict is None:
        mean_std_dict = {}
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            mean_std_dict[col] = {'mean': mean, 'std': std}
    else:
        for col in cols:
            mean = mean_std_dict[col]['mean']
            std = mean_std_dict[col]['std']
            df[col] = (df[col] - mean) / std
    return df, mean_std_dict



# 使用线性回归进行预测的函数
def predict_with_linear_regression(X, y, X_new):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # 计算残差标准差（Residual Standard Deviation）
    residuals = y_test - y_pred
    rss = np.sum(residuals**2)
    n = len(y_test)
    p = X_train.shape[1]
    rsd = np.sqrt(rss / (n - p - 1))

    # 为 X_train 添加截距项
    X_train_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

    # 计算新预测值的标准差
    X_new_with_intercept = np.hstack([np.ones((X_new.shape[0], 1)), X_new])  # 为 X_new 添加截距项
    inv_matrix = inv(X_train_with_intercept.T @ X_train_with_intercept)  # 使用包含截距项的 X_train
    var_pred = rsd ** 2 * (1 + np.diagonal(X_new_with_intercept @ inv_matrix @ X_new_with_intercept.T))
    std_pred = np.sqrt(var_pred)

    y_new_pred = linear_reg.predict(X_new)
    return y_new_pred, std_pred


def plot_weibull_density(y_preds, y_stds, features, X_new, mean_std_dict):
    plt.figure(figsize=(8, 5))

    # 对每个预测值进行反标准化
    y_pred_denorm = [y_pred * mean_std_dict[target]['std'] + mean_std_dict[target]['mean'] for y_pred in y_preds]

    # 如果 y_stds 是单一数字，则复制为列表，否则使用原列表
    y_std_denorm = [y_stds * mean_std_dict[target]['std']] * len(y_preds) if np.isscalar(y_stds) else y_stds

    # 检查哪些特征有不同的值
    unique_feature_values = {feat: X_new[feat].nunique() != 1 for feat in features}
    # 收集所有不同值的特征名称
    different_features = [feat for feat, is_different in unique_feature_values.items() if is_different]

    for i, (y_pred, y_std) in enumerate(zip(y_pred_denorm, y_std_denorm)):
        print(f"Plotting distribution for prediction {i + 1}")  # 确认正在处理的预测值

        # 使用固定的形状参数 k 和动态计算的尺度参数 lambda
        k = 4  # 根据数据选择合适的 k 值
        lambda_ = y_pred / (np.log(2) ** (1 / k))  # 使得中值对应于 y_pred

        # 生成 Weibull 分布的数据点
        x_values = np.linspace(weibull_min.ppf(0.001, k, scale=lambda_), weibull_min.ppf(0.999, k, scale=lambda_), 100)
        y_probs = weibull_min.pdf(x_values, k, scale=lambda_)

        # 仅为值不同的特征生成特征值字符串
        feature_values = ', '.join(
            [f'{feat}={X_new.iloc[i][feat]:.2f}' for feat in features if unique_feature_values[feat]])

        # 绘制 Weibull 分布
        plt.plot(x_values, y_probs, label=f'{feature_values}')
        plt.fill_between(x_values, y_probs, alpha=0.2)

    plt.title(f'Weibull PDF of Working Time Affected by {" & ".join(different_features)}')
    plt.xlabel('Working Time (Days)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


# 定义特征和目标变量
features = ['oversizing_rate', 'top_1_high_average', 'top_1_low_average','event_number', 'top_1_temp_diff', 'top_1_wind_average']
target = 'working_time'

# 标准化数据，并获取均值和标准差
data_standardized, mean_std_dict = standardize(data.copy(), features + [target])

# 指定两个固定值 X_new，以进行比较
X_new = pd.DataFrame({
    'oversizing_rate': [0.01, 0.01, 0.01],  # 两个不同的示例值
    'top_1_high_average': [32, 32, 32],  # 相同的示例值
    'top_1_low_average': [-22, -22, -22] , # 相同的示例值
    'event_number':     [0.02,0.02, 0.02],  # 两个不同的示例值
    'top_1_temp_diff':  [24,26, 28],   # 相同的示例值
    'top_1_wind_average': [33, 33,33]   # 相同的示例值
})

# 使用相同的均值和标准差来标准化 X_new
X_new_standardized, _ = standardize(X_new.copy(), features, mean_std_dict)

# 进行预测
# 用新的特征和目标调用该函数
y_new_pred, y_new_std = predict_with_linear_regression(data_standardized[features], data_standardized[target], X_new_standardized)

# 调用绘图函数，确保为 y_stds 提供与 y_preds 相同长度的列表
plot_weibull_density(y_new_pred, [y_new_std] * len(y_new_pred), features, X_new, mean_std_dict)
