## lifetime distribution kde, joint, weilbull
'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data_path = 'Total_Inverter_Results_Analysis4.csv'  # 确保这里的路径与你的数据文件路径相匹配
data = pd.read_csv(data_path)

# 确保所有数据是数值型，且移除任何NaN值
data.dropna(subset=['working_time', 'oversizing_rate'], inplace=True)
data['working_time'] = pd.to_numeric(data['working_time'], errors='coerce')
data['oversizing_rate'] = pd.to_numeric(data['oversizing_rate'], errors='coerce')
data = data.dropna()

# 计算核密度估计
data_points = data[['working_time', 'oversizing_rate']].values.T
kde = gaussian_kde(data_points)

# 创建网格上的点来评估KDE
working_time_min, working_time_max = data['working_time'].min(), data['working_time'].max()
oversizing_rate_min, oversizing_rate_max = data['oversizing_rate'].min(), data['oversizing_rate'].max()
working_time, oversizing_rate = np.mgrid[working_time_min:working_time_max:100j, oversizing_rate_min:oversizing_rate_max:100j]
positions = np.vstack([working_time.ravel(), oversizing_rate.ravel()])
density = np.reshape(kde(positions).T, working_time.shape)

# 绘制三维表面图
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(working_time, oversizing_rate, density, cmap='viridis', edgecolor='none')
ax.set_xlabel('Working Time')
ax.set_ylabel('Oversizing Rate')
ax.set_zlabel('Density')
ax.set_title('3D Surface Plot of Probability Density')
fig.colorbar(surf, shrink=0.5, aspect=5)  # 添加颜色条
plt.show()
'''



# joint pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data = pd.read_csv('Total_Inverter_Results_Analysis4.csv')

# 确保所有数据是数值型，且移除任何NaN值
data.dropna(subset=['working_time', 'oversizing_rate'], inplace=True)
data['working_time'] = pd.to_numeric(data['working_time'], errors='coerce')
data['oversizing_rate'] = pd.to_numeric(data['oversizing_rate'], errors='coerce')
data = data.dropna()

# 对每个变量分别进行正态分布参数估计
working_time_mean, working_time_std = norm.fit(data['working_time'])
oversizing_rate_mean, oversizing_rate_std = norm.fit(data['oversizing_rate'])

# 创建网格上的点来评估正态分布
working_time_values = np.linspace(data['working_time'].min(), data['working_time'].max(), 100)
oversizing_rate_values = np.linspace(data['oversizing_rate'].min(), data['oversizing_rate'].max(), 100)
working_time_grid, oversizing_rate_grid = np.meshgrid(working_time_values, oversizing_rate_values)

# 计算网格上每个点的正态分布PDF
pdf_working_time = norm(working_time_mean, working_time_std).pdf(working_time_grid)
pdf_oversizing_rate = norm(oversizing_rate_mean, oversizing_rate_std).pdf(oversizing_rate_grid)

# 假设独立性，计算联合PDF
joint_pdf = pdf_working_time * pdf_oversizing_rate

# 绘制三维表面图
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(working_time_grid, oversizing_rate_grid, joint_pdf, cmap='viridis', edgecolor='none')
ax.set_xlabel('Working Time')
ax.set_ylabel('Oversizing Rate')
ax.set_zlabel('Probability Density')
ax.set_title('3D Surface Plot of lifetime')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()




'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import weibull_min

data = pd.read_csv('Total_Inverter_Results_Analysis4.csv')

# # 如果你已经有了CSV文件，直接用 pd.read_csv() 加载你的数据即可
# data = pd.DataFrame({
#     'working_time': np.random.uniform(1000, 3000, 1000),  # 模拟数据
#     'oversizing_rate': np.random.uniform(0.0, 0.1, 1000)  # 模拟数据
# })

# 确保所有数据是数值型，且移除任何NaN值
data['working_time'] = pd.to_numeric(data['working_time'], errors='coerce')
data['oversizing_rate'] = pd.to_numeric(data['oversizing_rate'], errors='coerce')
data.dropna(subset=['working_time', 'oversizing_rate'], inplace=True)

# 对工作时间进行Weibull分布参数估计
params_wt = weibull_min.fit(data['working_time'], floc=0)

# 对超额配比率进行Weibull分布参数估计
params_or = weibull_min.fit(data['oversizing_rate'], floc=0)

# 创建值网格
wt_values = np.linspace(np.min(data['working_time']), np.max(data['working_time']), 100)
or_values = np.linspace(np.min(data['oversizing_rate']), np.max(data['oversizing_rate']), 100)
wt_grid, or_grid = np.meshgrid(wt_values, or_values)

# 计算网格上每个点的Weibull PDF
pdf_wt = weibull_min(*params_wt).pdf(wt_grid)
pdf_or = weibull_min(*params_or).pdf(or_grid)

# 计算联合PDF（因为我们假设两个变量是独立的）
joint_pdf = pdf_wt * pdf_or

# 绘制三维表面图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wt_grid, or_grid, joint_pdf, cmap='viridis', edgecolor='none')
ax.set_xlabel('Working Time')
ax.set_ylabel('Oversizing Rate')
ax.set_zlabel('Joint Probability Density')
ax.set_title('3D Surface Plot of Joint Weibull Probability Density')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
'''
