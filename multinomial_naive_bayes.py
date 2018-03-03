import numpy as np

from games import nb_train2, nb_predict2, error2


def read_matrix(file):
    fd = open(file, 'r')
    header = fd.readline()
    n_row, n_col = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((n_row, n_col))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    fd.close()
    return matrix, tokens, np.array(Y)


def nb_train(document_matrix, labels):
    parameters = dict()
    m = labels.size
    n_positives = np.sum(labels)
    docs_sizes = np.sum(document_matrix, axis=-1)
    p_y = (n_positives+1)/(m+2)
    p_denom = docs_sizes.dot(1-labels)+2
    q_denom = docs_sizes.dot(labels)+2
    p_xi = (document_matrix.T.dot(1-labels)+1)/p_denom
    q_xi = (document_matrix.T.dot(labels)+1)/q_denom
    parameters['p_doc_prior'] = p_y
    parameters['p_word_occur_pos_class'] = q_xi
    parameters['p_word_occur_zero_class'] = p_xi
    return parameters


def nb_compute_probabilities(parameters, test_doc_matrix):
    p_y = parameters['p_doc_prior']
    log_p_xi = np.log(parameters['p_word_occur_zero_class'])
    log_q_xi = np.log(parameters['p_word_occur_pos_class'])
    log_ratio = test_doc_matrix.dot(log_p_xi - log_q_xi) + np.log(1 - p_y) - np.log(p_y)
    probs_pos = 1/(1+np.exp(log_ratio))
    return probs_pos


def nb_predict(parameters, test_doc_matrix):
    probs_pos = nb_compute_probabilities(parameters, test_doc_matrix)
    probs_zeros = 1 - probs_pos
    evals = np.vstack((probs_zeros, probs_pos))
    preds = np.argmax(evals, axis=0)
    return preds


def error(validation_labels, output_labels):
    acc = np.sum(np.abs(validation_labels-output_labels))/validation_labels.size
    return acc


def get_indicative_tokens(p_word_occur_zero_class, p_word_occur_pos_class, tokens):
    indices = np.argsort(np.log(p_word_occur_pos_class) - np.log(p_word_occur_zero_class))[::-1]
    res = []
    for i in range(5):
        res.append(tokens[indices[i]])
    return res
#
# def multinomial_log(x, p):
#     p = np.log(p)
#     log_factorials = {}
#     num_trials = int(np.sum(x))
#     x_probability = 0
#     for i in range(1, num_trials+1):
#         x_probability += np.log(i)
#         log_factorials[i] = x_probability
#     indices = np.where(x > 1)
#     for i in indices[0]:
#         x_probability -= log_factorials[int(x[i])]
#     x_probability += x.dot(p)
#     return x_probability
#


def main():
    train_files = ('MATRIX.TRAIN.50', 'MATRIX.TRAIN.100', 'MATRIX.TRAIN.200',
                      'MATRIX.TRAIN.400', 'MATRIX.TRAIN.800', 'MATRIX.TRAIN.1400', 'MATRIX.TRAIN')
    test_matrix, tokens, test_labels = read_matrix('MATRIX.TEST')
    for file in train_files:
        train_matrix, tokens, train_labels = read_matrix(file)
        params = nb_train(train_matrix, train_labels)

        predictions = nb_predict(params, test_matrix)
        print(file+' : Error: %1.4f' % error(test_labels, predictions))
    train_matrix, tokens, train_labels = read_matrix('MATRIX.TRAIN')
    params = nb_train(train_matrix, train_labels)
    tokens = get_indicative_tokens(params['p_word_occur_zero_class'], params['p_word_occur_pos_class'], tokens)
    print(tokens)

if __name__ == '__main__':
    main()

