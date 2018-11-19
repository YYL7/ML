# Naive Bayes

import numpy as np

"""
Naive Bayes:
p(xy)=p(x|y)p(y)=p(y|x)p(x)
p(x|y)=p(y|x)p(x)/p(y)
"""




# load the dataset
# using Naive Bayes to determine if the text is insultuing or not
def load_data_set():
    """
    posting_list: word list
    class_vec: classification
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 means Insulting text, 0 is not
    class_vec = [0, 1, 0, 1, 0, 1]  
    return posting_list, class_vec

  
  
  # create vacabulary list
  def create_vocab_list(data_set):
    """
    Get a set of all words
    data_set: dataset
    return: a set of all words without repeating elements
    """
    # create empty set
    vocab_set = set()  
    for item in data_set:
        # | union for two sets
        # set: without repeating elements
        vocab_set = vocab_set | set(item)  
    return list(vocab_set)


  
def set_of_words2vec(vocab_list, input_set):
    """
    For the whole dataset,to see if the word appears, if yes then set 1
    vocab_list: List of all words
    input_set: input dataset
    return: get[0,1,0,1...]，1 means the words in the vocabulary appear in the input dataset. 0 means no
    """
    # Create a vector of the same length as the vocab_listand set all elements to 0
    result = [0] * len(vocab_list)
    # For the whole dataset,to see if the word appears, if yes then set 1
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            # print('the word: {} is not in my vocabulary'.format(word))
            pass
    return result

  
  
  
  def train_naive_bayes(train_mat, train_category):
    """
    train_mat:  type is ndarray, [[0,1,0,1], [], []]
    train_category: classification for the training dataset， [0, 1, 0],
                    
    """
    # The length of the list should be equal to the length of the train_mat
    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])
    # Because insulting word is marked as 1, so you can get insulting as long as you add them together.
    # the frequency of the insulting word，is the sum of the 1 in train_category
    # divide by by number of the document, we can  get the probability of the insulting file
    pos_abusive = np.sum(train_category) / train_doc_num
    # Number of times a word appears
    # set to one instead of 0, to prevent overflow if the number is too small
    p0num = np.ones(words_num) # category 0
    p1num = np.ones(words_num) # category 1
    # The number of times the entire dataset word appears
    p0num_all = 2.0 # category 0
    p1num_all = 2.0 # category 1

    for i in range(train_doc_num):
        # Iterate through all the files, and if it is an insulting file, 
        # count the number of insulting words that appear in this insulting file.
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])
    # log function 
    p1vec = np.log(p1num / p1num_all) # probability of the category 1
    p0vec = np.log(p0num / p0num_all) # # probability of the category 0
    return p0vec, p1vec, pos_abusive
  
  
 


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):
    """
        # Convert multiplication to addition
        Multiply：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        Add：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    vec2classify: training dataset, [0,1,1,1,1...]，hte vector to be classified
    p0vec: catogory 0，[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]
    p1vec: catogory 1，[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]
    p_class1: probability of the catogory 1
    return: catogory of 1 or 0
    """
    # The probability of the category under the condition of a word in the vocabulary
    p1 = np.sum(vec2classify * p1vec) + np.log(p_class1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


      
def bag_words2vec(vocab_list, input_set):
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1
        else:
            print('the word: {} is not in my vocabulary'.format(word))
    return result      
      
  
 


# testing
def testing_naive_bayes():
    # 1. load the dataset
    list_post, list_classes = load_data_set()
    # 2. create the list of vocalbulary
    vocab_list = create_vocab_list(list_post)

    # 3. Determine whether a word appears and create a matrix
    train_mat = []
    for post_in in list_post:
        # return matrix of  m*len(vocab_list) 
        train_mat.append(set_of_words2vec(vocab_list, post_in))
    # 4. training dataset
    p0v, p1v, p_abusive = train_naive_bayes(np.array(train_mat), np.array(list_classes))
    # 5. testing dataset
    test_one = ['love', 'my', 'dalmation']
    test_one_doc = np.array(set_of_words2vec(vocab_list, test_one))
    print('the result is: {}'.format(classify_naive_bayes(test_one_doc, p0v, p1v, p_abusive)))
    test_two = ['stupid', 'garbage']
    test_two_doc = np.array(set_of_words2vec(vocab_list, test_two))
    print('the result is: {}'.format(classify_naive_bayes(test_two_doc, p0v, p1v, p_abusive)))


