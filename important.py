import statsmodels.api as sm
import pandas as pd
import numpy as np
import csv
import sys

num = sys.argv[1]
# 使用 pandas 读取 CSV 文件
doe_data_ori = pd.read_csv('all_{}.csv'.format(num))

design = [[-1, -1, -1, 1, 1, 1, -1],
          [-1, -1, 1, 1, -1, -1, 1],
          [-1, 1, -1, -1, 1, -1, 1],
          [-1, 1, 1, -1, -1, 1, -1],
          [1, -1, -1, -1, -1, 1, 1],
          [1, -1, 1, -1, 1, -1, -1],
          [1, 1, -1, 1, -1, -1, -1],
          [1, 1, 1, 1, 1, 1, 1]
        ]
'''
design = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
        ]
'''
#doe_data = doe_data_ori.drop(columns=['binary'])

#doe_data = doe_data[doe_data.mean(axis=1) >= 2]
#对于编译或者运行出错的程序，可以把它当成运行时间无线大来处理
doe_data = doe_data_ori.fillna(float('inf'))
#不排除错误的选项，这个文件可以留着找错误选项用
#doe_data.to_csv('doe_data_clean_{}.csv'.format(num),index=True)
#doe_data.reset_index(inplace = True, drop = True)
doe_data = doe_data[~doe_data.isin([float('inf')]).any(axis=1)]
#valid_columns = doe_data.select_dtypes(include='number').columns
print(doe_data.columns[1:9])
#print("valid_columns = ",valid_columns)
doe_data['平均值'] = doe_data[doe_data.columns[1:9]].mean(axis=1)
doe_data['标准差比率'] = (doe_data[doe_data.columns[1:9]].std(axis=1) / doe_data[doe_data.columns[1:9]].mean(axis=1))*100
#排除错误项，
doe_data.to_csv('doe_data_{}.csv'.format(num),index=True, float_format='%.4f')
X = sm.add_constant(design)

result_df = pd.DataFrame(columns=['binary_name'] + [f'sorted_{i}' for i in range(7)])
for index, row in doe_data.iterrows():
    #print("index = ",index)
    binary_name = doe_data_ori.loc[index,'binary']
    #print("binary_name = ",binary_name)
    # 处理每一行数据
    Y = np.array(row[1:9])
    #print("Y = ",Y)
    model = sm.OLS(Y,X)
    result = model.fit()
    coeff = result.params[1:]
    #print(coeff)
    # 使用 `enumerate()` 函数和 lambda 表达式创建数字对
    indexed_coeff = [(i, c) for i, c in enumerate(coeff)]
    # 按 `coeff` 的值从小到大进行排序
    sorted_coeff = sorted(indexed_coeff, key=lambda x: x[1])
    #print("len = ",len(sorted_coeff))
    #result_df = result_df.append({'binary_name': binary_name, 'sorted_coeff': sorted_coeff}, ignore_index=True)
    row_data = {'binary_name': binary_name}
    for i in range(7):
        #print(i)
        row_data[f'sorted_{i}'] = sorted_coeff[i]
    result_df = result_df.append(row_data, ignore_index=True)

# 将结果写入 CSV 文件
result_df.to_csv('importance_{}.csv'.format(num), index=True)
