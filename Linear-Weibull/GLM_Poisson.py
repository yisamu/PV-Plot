#############Generalized Linear Models###############
######################Poisson  log link function############

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import weibull_min



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

def check_and_adjust_data(X):
    X_adj = X.copy()
    # 仅选择数值类型的列进行操作
    numeric_cols = X_adj.select_dtypes(include=[np.number]).columns
    # 对这些数值列，将所有小于或等于0的值调整为一个较小的正数
    X_adj[numeric_cols] = X_adj[numeric_cols].applymap(lambda x: 0.01 if x <= 0 else x)
    return X_adj


def predict_with_glm_and_std(X, y, X_new):
    print("X_new before adding constant:\n", X_new)

    # 确保为X_new添加常数项，并正确赋值
    X_new_with_intercept = sm.add_constant(X_new, has_constant='add')
    print("X_new_with_intercept:\n", X_new_with_intercept)

    X_with_intercept = sm.add_constant(X, has_constant='add')

    # 使用泊松家族拟合GLM模型
    glm = sm.GLM(y, X_with_intercept, family=sm.families.Poisson())
    glm_results = glm.fit()

    # 使用模型进行预测
    y_new_pred = glm_results.predict(X_new_with_intercept)

    # 计算标准差
    std_pred = np.sqrt(y_new_pred)  # 假设使用泊松分布的方差作为标准差的近似

    return y_new_pred, std_pred


def plot_weibull_density(y_preds, y_stds, features, X_new, mean_std_dict):
    plt.figure(figsize=(8, 5))
    y_pred_denorm = [y_pred * mean_std_dict[target]['std'] + mean_std_dict[target]['mean'] for y_pred in y_preds]
    y_std_denorm = [y_stds * mean_std_dict[target]['std']] * len(y_preds) if np.isscalar(y_stds) else y_stds

    unique_feature_values = {feat: X_new[feat].nunique() != 1 for feat in features}
    different_features = [feat for feat, is_different in unique_feature_values.items() if is_different]

    for i, (y_pred, y_std) in enumerate(zip(y_pred_denorm, y_std_denorm)):
        print(f"Plotting distribution for prediction {i + 1}")
        k = 4
        lambda_ = y_pred / (np.log(2) ** (1 / k))
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
    features = ['oversizing_rate', 'top_1_high_average', 'top_1_low_average', 'event_number', 'top_1_temp_diff', 'top_1_wind_average']
    target = 'working_time'

    data_standardized, mean_std_dict = standardize(data.copy(), features + [target])

    X_new = pd.DataFrame({
        'oversizing_rate': [0.01, 0.01, 0.01],
        'top_1_high_average': [32, 32, 32],
        'top_1_low_average': [-20, -28, -30],
        'event_number': [0.02, 0.02, 0.02],
        'top_1_temp_diff': [24, 24, 24],
        'top_1_wind_average': [33, 33, 33]
    })
    # 调整特征数据
    X_adjusted = check_and_adjust_data(data_standardized[features])
    # 标准化新数据
    X_new_standardized, _ = standardize(X_new.copy(), features, mean_std_dict)
    # 调整新数据集
    X_new_adjusted = check_and_adjust_data(X_new_standardized)

    # 应用数据调整
    data_standardized_adj = check_and_adjust_data(data_standardized)
    X_new_standardized_adj = check_and_adjust_data(X_new_standardized)

    # 进行预测
    y_new_pred, y_new_std = predict_with_glm_and_std(data_standardized_adj[features], data_standardized_adj[target],
                                                     X_new_standardized_adj)
    # 绘制 Weibull 分布密度图
    plot_weibull_density(y_new_pred, [y_new_std] * len(y_new_pred), features, X_new, mean_std_dict)