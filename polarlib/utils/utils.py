import json, numpy

def convert_sentiment_attitude(sentiment_value, sentiment_mapping):

    sentiment_category = "NEUTRAL"

    for category, category_bins in sentiment_mapping.items():

        for b in category_bins:

            if sentiment_value >= b[0] and sentiment_value < b[1]:
                sentiment_category = category
                break

    return sentiment_category

def load_article(path, func=lambda t: json.loads(json.load(t))):
    """
    Load an article from a JSON file.

    Args:
        path (str): Path to the JSON file containing the article data.
        func (function, optional): A function to process the file object. Default is to use json.loads(json.load(t)).

    Returns:
        dict: The loaded article data as a dictionary.
    """
    with open(path, 'r') as f:

        article_obj = func(f)

    return article_obj


def jaccard_index(s1, s2):
    """
    Calculate the Jaccard index between two sets.

    Args:
        s1 (set or list): First set or list.
        s2 (set or list): Second set or list.

    Returns:
        float: Jaccard index value between the two sets.
    """
    return len(set(s1).intersection(set(s2))) / len(set(s1).union(set(s2)))


def to_chunks(lst, n):
    """
    Split a list into chunks of a specified size.

    Args:
        lst (list): The input list to be split.
        n (int): The size of each chunk.

    Yields:
        list: A chunk of the input list with the specified size.
    """
    for i in range(0, len(lst), n):

        yield lst[i:i + n]


def is_subsequence(sub, seq):
    """
    Check if a list is a subsequence of another list.

    Args:
        sub (list): The potential subsequence.
        seq (list): The sequence to check.

    Returns:
        bool: True if 'sub' is a subsequence of 'seq', False otherwise.
    """
    it = iter(seq)

    return all(c in it for c in sub)


def find_longest_unique_subsequences(input_list):
    """
    Find the longest unique subsequences in a list of lists.

    Args:
        input_list (list): A list containing lists as elements.

    Returns:
        list: List of longest unique subsequences from the input list.
    """
    input_list.sort(key=len, reverse=True)
    unique_subsequences = []

    for sublist in input_list:
        if not any(is_subsequence(sublist, existing) for existing in unique_subsequences):
            unique_subsequences.append(sublist)

    return unique_subsequences


def sentiment_threshold_difference(swn_pos, swn_neg):
    """
    Calculate the sentiment threshold difference using SentiWordNet scores.

    Args:
        swn_pos (float): Positive sentiment score from SentiWordNet.
        swn_neg (float): Negative sentiment score from SentiWordNet.

    Returns:
        float: Sentiment threshold difference value.
    """
    swn_pos = abs(swn_pos)
    swn_neg = abs(swn_neg)
    return numpy.sign(swn_pos - swn_neg) * (abs(swn_pos - swn_neg))

def calculate_value_buckets(values, verbose=False):

    if verbose:

        print('Mean:', numpy.mean(values))
        print('Std.:', numpy.std(values))
        print()

    n, bins = numpy.histogram(values)

    if verbose:

        for k, ij in enumerate(zip(bins[:-1], bins[1:])):

            i = ij[0]
            j = ij[1]

            print(f'>= {i:<25} and < {j:<25}:', n[k])

    bins = list(zip(bins[:-1], bins[1:]))

    return bins