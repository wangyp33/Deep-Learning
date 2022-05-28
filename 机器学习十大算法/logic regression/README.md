逻辑回归模型

lr = LogisticRegression(C=1e5)  
lr.fit(X,Y)
初始化逻辑回归模型并进行训练，C=1e5表示目标函数。

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
获取的鸢尾花两列数据，对应为花萼长度和花萼宽度，每个点的坐标就是(x,y)。 先取X二维数组的第一列（长度）的最小值、最大值和步长h（设置为0.02）生成数组，再取X二维数组的第二列（宽度）的最小值、最大值和步长h生成数组， 最后用meshgrid函数生成两个网格矩阵xx和yy。


Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
调用ravel()函数将xx和yy的两个矩阵转变成一维数组，由于两个矩阵大小相等，因此两个一维数组大小也相等。np.c_[xx.ravel(), yy.ravel()]是获取矩阵
