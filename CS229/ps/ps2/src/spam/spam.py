import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    message = message.lower()
    words = message.split(" ")
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    count_words_vs_messages = collections.Counter()
    word_dict = dict()

    for message in messages:
        words = get_words(message)
        unique_words = set(words)
        for word in unique_words:
            count_words_vs_messages[word] += 1

    vocab = [k for k, v in count_words_vs_messages.items() if v >= 5]
    indices = list(range(0, len(vocab)))
    word_dict = dict(zip(vocab, indices))
          
    return word_dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    transformed_text = np.zeros([len(messages), len(word_dictionary)])

    for msg_idx, msg in enumerate(messages):
        # print(msg)
        word_counts = collections.Counter(get_words(msg))
        for word, count in word_counts.items():
            if word in word_dictionary:
                col_idx = word_dictionary[word]
                transformed_text[msg_idx, col_idx] = count
                # print(f'key: {word}, count: {count}, col_idx: {col_idx}')
    return transformed_text
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    # Count labels, evaluate phi_y
    count_y1 = len(labels[labels == 1])
    phi = count_y1 / len(labels)

    # Count all words when labels = 1 or 0
    msg_lengths = np.sum(matrix,axis=1) # (4457, 1)
    sum_all_words_and_y1 = sum(msg_lengths[labels==1]) # 11134.0
    sum_all_words_and_y0 = sum(msg_lengths) - sum_all_words_and_y1 # 44796. If summed = 55930
    
    # Count specific words when labels = 1 or 0
    word_count_and_y1 = np.sum(matrix[labels==1], axis=0) # if summed = 11134.0
    word_count_and_y0 = np.sum(matrix[labels==0], axis=0) # if summed = 44796.0. Total is 55930

    # Evaluate phi_k|y, conditional probability of specific word when label = 1 or 0
    # With laplace smooth. See p14 CS229 lectures notes 2
    vocab_length = matrix.shape[1] # 1758
    phi_y1 = (word_count_and_y1 + 1) / (sum_all_words_and_y1 + vocab_length)
    phi_y0 = (word_count_and_y0 + 1)/ (sum_all_words_and_y0 + vocab_length)
    print(phi, phi_y1, phi_y0)

    return phi, phi_y0, phi_y1
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi, phi_y0, phi_y1 = model
    
    prob_y0 = np.log(1 - phi) + np.sum(matrix * np.log(phi_y0), axis=1)
    prob_y1 = np.log(phi) + np.sum(matrix * np.log(phi_y1), axis=1)

    y_predicted = (prob_y1 > prob_y0).astype(np.int64)
    return y_predicted 
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    __, phi_y0, phi_y1 = model
    inv_dictionary = {v: k for k, v in dictionary.items()}

    spam_strength = np.log(phi_y1 / phi_y0)
    sort_idx = np.argsort(spam_strength) # sorts lowest to highest
    top5_idx = sort_idx[-5:] # get last 5 elements

    top5_spam_words = [inv_dictionary[idx] for idx in top5_idx]
    return top5_spam_words
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_accuracy = -np.inf
    best_radii = None
    for radii in radius_to_consider:
        svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radii)
        svm_accuracy = np.mean(svm_predictions == val_labels)
        if svm_accuracy > best_accuracy:
            best_accuracy = svm_accuracy
            best_radii = radii

    return best_radii

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:], fmt='%s')

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels) 
    # predicts 65 spam, True spam is 67, 0.978494623655914

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
