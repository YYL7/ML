# Logistic Regression

import numpy as np

# to load the dataset
def load_data_set():
    """
    to load the dataset
    :return:return two arrays (data, lable)
        data_arr -- features
        label_arr -- lables/classcification
    """
    data_arr = []
    label_arr = []
    f = open('db/5.Logistic/TestSet.txt', 'r')
    for line in f.readlines():
        line_arr = line.strip().split()
        data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr, label_arr

# sigmoid fuction
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# Gradient Ascent
# Gradient Ascent uses the whole dataset on each update.
def grad_ascent(data_arr, class_labels):
    """
    :param class_labels: class_labels, a row vector with 1*100, so we need to transpose the row vector to column vector
    :return: 
    """
    # turn the data_arr to numpy matrix
    data_mat = np.mat(data_arr)
    # transpose
    label_mat = np.mat(class_labels).transpose()
    # m->row(length of the data), n->nummber of features
    m, n = np.shape(data_mat)
    # learning rate
    alpha = 0.001
    # max cycles
    max_cycles = 500
    # weights 
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        # m x 3 dot 3 x 1
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        weights = weights + alpha * data_mat.transpose() * error
    return weights

# data visualization
def plot_best_fit(weights):
    """
    plot
    :param weights: 
    :return: 
    """
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    """
    how do we get y?
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

# Stochastiic Gradient Ascent
# stochastiic gradient ascent, only use one sample on each update.
def stoc_grad_ascent0(data_mat, class_labels):
    """
    :param data_mat: features（except the last column）,ndarray
    :param class_labels: labels（data from last column）
    :return: the best weights
    """
    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # sum(data_mat[i]*weights) to get f(x)， f(x)=a1*x1+b2*x2+..+nn*xn,
        # h is a value, not a matrix
        h = sigmoid(sum(data_mat[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights

# to improve the Stochastiic Gradient Ascent
def stoc_grad_ascent1(data_mat, class_labels, num_iter=150):
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        # use list here
        data_index = list(range(m))
        for i in range(m):
            # i and j get larger，thus alpha decrease, but not equal to zero
            alpha = 4 / (1.0 + j + i) + 0.01
            # get a random number 
            # random.uniform(x, y) get a random number，which is in the range of [x,y]
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
            error = class_labels[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mat[data_index[rand_index]]
            del(data_index[rand_index])
    return weights

# Test
def test():
    data_arr, class_labels = load_data_set()
    # Note:，grad_ascent return a matrix, so use getA() to get ndarray
    # weights = grad_ascent(data_arr, class_labels).getA()
    # weights = stoc_grad_ascent0(np.array(data_arr), class_labels)
    weights = stoc_grad_ascent1(np.array(data_arr), class_labels)
    plot_best_fit(weights)



