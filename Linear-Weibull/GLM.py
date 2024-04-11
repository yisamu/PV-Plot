#############Generalized Linear Models###############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import statsmodels.api as sm

data = pd.read_csv('C:/Users/yandcuo/Box/Dr/project/process/data/SunEnergy1_DataSet/2024/03/Total_Inverter_Results_Analysis4.csv')


# 数据标准化函数，处理 NaN 值
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

# 广义线性模型预测及标准差计算
#正态分布
def predict_with_glm_and_std(X, y, X_new):
    # 打印 X_new 查看其结构和数据类型
    print("X_new before adding constant:\n", X_new)

    # 手动添加截距项
    X_new_with_intercept = X_new.copy()
    X_new_with_intercept['intercept'] = 1.0  # 添加截距项
    X_new_with_intercept = X_new_with_intercept[['intercept'] + X_new.columns.tolist()]
    print("X_new_with_intercept:\n", X_new_with_intercept)

    # 为 X 添加截距项
    X_with_intercept = sm.add_constant(X)

    # 使用 GLM 拟合模型
    glm = sm.GLM(y, X_with_intercept, family=sm.families.Gaussian())
    glm_results = glm.fit()

    # 预测新值
    y_new_pred = glm_results.predict(X_new_with_intercept)

    # 计算预测标准差
    cov_params = glm_results.cov_params()
    cov_params_reindexed = cov_params.reindex(index=X_new_with_intercept.columns,
                                              columns=X_new_with_intercept.columns, fill_value=0)
    var_pred = (X_new_with_intercept.dot(cov_params_reindexed) * X_new_with_intercept).sum(1)
    std_pred = np.sqrt(var_pred)

    return y_new_pred, std_pred

# 绘制 Weibull 分布密度图
def plot_weibull_density(y_preds, y_stds, features, X_new, mean_std_dict):
    plt.figure(figsize=(8, 5))

    # 反标准化预测值
    y_pred_denorm = [y_pred * mean_std_dict[target]['std'] + mean_std_dict[target]['mean'] for y_pred in y_preds]
    y_std_denorm = [y_stds * mean_std_dict[target]['std']] * len(y_preds) if np.isscalar(y_stds) else y_stds

    # 确定不同特征值的特征名称
    unique_feature_values = {feat: X_new[feat].nunique() != 1 for feat in features}
    different_features = [feat for feat, is_different in unique_feature_values.items() if is_different]

    # 绘制 Weibull 分布
    for i, (y_pred, y_std) in enumerate(zip(y_pred_denorm, y_std_denorm)):
        print(f"Plotting distribution for prediction {i + 1}")

        k = 4  # 形状参数
        lambda_ = y_pred / (np.log(2) ** (1 / k))  # 尺度参数
        x_values = np.linspace(weibull_min.ppf(0.001, k, scale=lambda_), weibull_min.ppf(0.999, k, scale=lambda_), 100)
        y_probs = weibull_min.pdf(x_values, k, scale=lambda_)

        feature_values = ', '.join([f'{feat}={X_new.iloc[i][feat]:.2f}' for feat in features if unique_feature_values[feat]])
        plt.plot(x_values, y_probs, label=f'{feature_values}')
        plt.fill_between(x_values, y_probs, alpha=0.2)

    plt.title(f'Weibull PDF of Working Time Affected by {" & ".join(different_features)}')
    plt.xlabel('Working Time (Days)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 定义特征和目标变量
    features = ['oversizing_rate', 'top_1_high_average', 'top_1_low_average', 'event_number', 'top_1_temp_diff', 'top_1_wind_average']
    target = 'working_time'

    # 标准化数据，并获取均值和标准差
    data_standardized, mean_std_dict = standardize(data.copy(), features + [target])

    # 指定新数据进行比较
    X_new = pd.DataFrame({
        'oversizing_rate': [0.01, 0.01, 0.01],
        'top_1_high_average': [32, 32, 32],
        'top_1_low_average': [-22, -26, -30],
        'event_number': [0.02, 0.02, 0.02],
        'top_1_temp_diff': [24, 24, 24],
        'top_1_wind_average': [33, 33, 33]
    })

    # 标准化新数据
    X_new_standardized, _ = standardize(X_new.copy(), features, mean_std_dict)

    # 进行预测
    y_new_pred, y_new_std = predict_with_glm_and_std(data_standardized[features], data_standardized[target], X_new_standardized)

    # 绘制 Weibull 分布密度图
    plot_weibull_density(y_new_pred, [y_new_std] * len(y_new_pred), features, X_new, mean_std_dict)