# AdaBoost

import numpy as np

# load the simple dataset
def load_sim_data():
    """
    :return: data_arr   :features 
            label_arr   :lables
    """
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels

# load the dataset
def load_data_set(file_name):
    """
    :param file_name
    :return: np.array or np.matrix
    """
    num_feat = len(open(file_name).readline().split('\t'))
    data_arr = []
    label_arr = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))
    return np.matrix(data_arr), label_arr


def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    # data_mat[:, dimen], all values in the 'dimen' column of the dataset
    # thresh_ineq == 'lt',modify the value on the left. gt--on the right
    if thresh_ineq == 'lt':
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_array

# build the model
def build_stump(data_arr, class_labels, D):
    """
    :param data_arr: features
    :param class_labels: lables
    :param D: Initial weight value for the features
    :return: bestStump    :best model
            min_error     :error rate
            best_class_est  :Trained result set
    """
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T

    m, n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    # infinity
    min_err = np.inf
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                # Here is the matrix multiplier
                weighted_err = D.T * err_arr
                '''
                dim:             column of the features
                thresh_val:      threshold value
                inequal:         The error rate of being reversed left and right
                weighted_error:  The overall error rate
                best_class_est:  Optimal result of prediction (compare to class_labels）
                '''
                # print('split: dim {}, thresh {}, thresh inequal: {}, the weighted err is {}'.format(i, thresh_val, inequal, weighted_err))
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_err, best_class_est


def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    """
    :param data_arr: features
    :param class_labels: lables
    :param num_it: iteration times
    :return: weak_class_arr: weal classifier  
            agg_class_est: Predicted classification result 
    """
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    # initialize D--set weighted value for each feature, divide by m
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        # get the model
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        # print('D: {}'.format(D.T))

        # calculate alpha 
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        # store Stump Params in Array
        weak_class_arr.append(best_stump)
        # print('class_est: {}'.format(class_est.T))
        # Correct classification：get 1，will not change the result
        # Wrong Classification ：get -1，will change the result, so multiply by 1 
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
    
        # Calculate the e to the power of expon, and get the comprehensive probability
        # For the wrong sample, the sample weight value for D will become larger.
        # multiply by corresponding items
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # aggregate by the previous result
        agg_class_est += alpha * class_est
        # Sign: the positive is 1, 0 is 0, negative is -1, and get the symbol by the final weighted value.
        # Result：Wrong sample set, !=
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T,
                                 np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        # print('total error: {}\n'.format(error_rate))
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est

# Predicting through the weak classifiers above
def ada_classify(data_to_class, classifier_arr):
    """
    :param data_to_class: dataset
    :param classifier_arr: classifeir
    :return: +1 or -1, result for the classification
    """
    data_mat = np.mat(data_to_class)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(
            data_mat, classifier_arr[i]['dim'],
            classifier_arr[i]['thresh'],
            classifier_arr[i]['ineq']
        )
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)
    return np.sign(agg_class_est)

# Plot the ROC and calculate the area of AUC
def plot_roc(pred_strengths, class_labels):
    """
    :param pred_strengths: Weight value of the final prediction result
    :param class_labels: Classification result set of raw data
    :return: 
    """
    import matplotlib.pyplot as plt
    # variable to calculate AUC
    y_sum = 0.0
    # Sum of correct samples
    num_pos_class = np.sum(np.array(class_labels) == 1.0)
    # probability of the correct sample
    y_step = 1 / float(num_pos_class)
    # probability of the wrong sample
    x_step = 1 / float(len(class_labels) - num_pos_class)
    # np.argsort: The function returns the index value of the array from small to large.
    # get sorted index, it's reverse
    sorted_indicies = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # cursor
    cur = (1.0, 1.0)
    # loop through all the values, drawing a line segment at each point
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # draw line from cur to (cur[0]-delX, cur[1]-delY)
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    # diagonal line
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    # range (x1, x2, y1, y2)
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", y_sum * x_step)

# test
def test():
    # D = np.mat(np.ones((5, 1)) / 5)
    # data_mat, class_labels = load_sim_data()
    # print(data_mat.shape)
    # result = build_stump(data_mat, class_labels, D)
    # print(result)
    # classifier_array, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 9)
    # print(classifier_array, agg_class_est)
    data_mat, class_labels = load_data_set('7.AdaBoost/horseColicTraining2.txt')
    print(data_mat.shape, len(class_labels))
    weak_class_arr, agg_class_est = ada_boost_train_ds(data_mat, class_labels, 40)
    print(weak_class_arr, '\n-----\n', agg_class_est.T)
    plot_roc(agg_class_est, class_labels)
    data_arr_test, label_arr_test = load_data_set("7.AdaBoost/horseColicTest2.txt")
    m = np.shape(data_arr_test)[0]
    predicting10 = ada_classify(data_arr_test, weak_class_arr)
    err_arr = np.mat(np.ones((m, 1)))
    # test：Calculate the total number of samples, the number of error samples, the error rate
    print(m,
          err_arr[predicting10 != np.mat(label_arr_test).T].sum(),
          err_arr[predicting10 != np.mat(label_arr_test).T].sum() / m
          )


if __name__ == '__main__':
    test()
