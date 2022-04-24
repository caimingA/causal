from csv import writer
import pandas as pd
import numpy as np
# np.set_printoptions(threshold = np.inf)

from sklearn.datasets import load_boston

import graphviz
import lingam
# from lingam.utils import make_dot

print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

np.random.seed(100)

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)

df['PRICE'] = boston.target


model = lingam.DirectLiNGAM()
model.fit(df)

# print(model.adjacency_matrix_)
idx = np.abs(model.adjacency_matrix_) > 0.5

# print(idx)
dirs = np.where(idx) # 返回数组元素为true的值的横纵坐标, 因果关系 横坐标（dirs[1]） -> 纵坐标（dirs[0]）
# print(dirs)


labels = ['{}. {}'.format(i, col) for i, col in enumerate(df.columns)]

# names = labels if labels else ['{}'.format(i) for i in range(len(model.adjacency_matrix_))]

print(model.adjacency_matrix_[idx])


dot = graphviz.Digraph(format='png', engine='dot')

# 对于model.adjacency_matrix_[idx]来说，只打印true的值
for to, from_, coef in zip(dirs[0], dirs[1], model.adjacency_matrix_[idx]):
    print(to, " + ", from_, " + ", coef)
    dot.edge(labels[from_], labels[to], label=f'{coef:.2f}')


dot.view()

# 保存matrix和idx


writer = pd.ExcelWriter('Matrix.xlsx')
data = pd.DataFrame(model.adjacency_matrix_)
data.to_excel(writer, 'matrix', float_format='%.2f')
data = pd.DataFrame(idx)
data.to_excel(writer, 'idx', float_format='%.2f')

writer.save()

writer.close()

# # Save pdf
# dot.render('dag')

# # Save png
# dot.format = 'png'
# dot.render('dag')

# dot.view()
