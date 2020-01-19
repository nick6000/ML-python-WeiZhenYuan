# 鸢尾花（Iris Flower）分类
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
import pydotplus
import os
# 导入数据
filename = 'iris.data.csv'
name = {'separ-length', 'separ-width', 'petal-length', 'petal-width', 'class'}
dataset = read_csv(filename, names=name)

# 显示数据维度
print('数据维度:行%s,列%s'%dataset.shape)

# 查看数据前10行
print(dataset.head(10))

# 统计描述数据信息
print(dataset.describe())

# 分类分布情况
print(dataset.groupby('class').size())

# 箱线图
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# 直方图
dataset.hist()
pyplot.show()

# 散点矩阵图
scatter_matrix(dataset)
pyplot.show()

# 分离数据集
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7
X_train, X_validation, y_trian, y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 算法审查
models = {}
models['LR'] = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=3000)
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC(gamma='auto')

# 评估算法
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(models[key], X_train, y_trian, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s:%f(%f)' %(key, cv_results.mean(), cv_results.std()))
# 决策树模型训练
models['CART'].fit(X=X_train, y=y_trian)

# 决策树图形化
dot_data = export_graphviz(models['CART'], out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
path = os.getcwd() + '/'
tree_file = path + 'Iris.png'
try:
    os.remove(tree_file)
except:
    print('There is no file to be deleted.')
finally:
    graph.write(tree_file, format = 'png')

# 显示图像
image_data = imread(tree_file)
pyplot.imshow(image_data)
pyplot.axis('off')
pyplot.show()

# 评估算法
predictions = models['CART'].predict(X_validation)
print(accuracy_score(y_validation, predictions))