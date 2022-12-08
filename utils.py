from statistics import mean
import os
from contextlib import contextmanager
from sklearn import svm

def get_bigrams(word:str) -> list:
    return [i+j for i, j in \
            zip(word, word[1:])]




def pronounceable_score_heuristic(letters: str) -> float:
    """ Generates a numerical score

    Args:
        letters (str): string of length 2 containing only alphabetical characters to check our dataset for occurrences of

    Returns:
        float: a score representing the likelihood that we can pronounce this string of letters. 0.5 is generally pronounceable, 0.2 is not.
    """
    assert len(letters) == 2
    # check dataset for occurrences of [letters].
    # If they never appear, the string almost certainly cannot be pronounced
    # If they appear, determine how often by dividing the number of times they were found by the amount of words checked
    proportion = dict(in_line=0, not_in=0)

    #open dataset
    data_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mhyph.txt'))
    with open(data_path, 'r') as f:
        raw_text = f.readlines()
        for line in raw_text:
            if letters.lower() in line.lower():
                # we want to boost the weight of strings of letters that occur
                # more times in single words, since those are usually more easily 
                # pronounceable
                proportion['in_line'] += line.strip().count(letters)
            else:
                proportion['not_in'] += 1

        proportion['in_line'] -= 1 if (not proportion['in_line'] > 0) else 0
        # if the set of letters is never found, then it almost certainly can't be pronounced, or possibly is simply not in our dataset.
    # return the amount of times it was found divided by the total lines in the file (multiply by 10 to trim leading zeroes)
    return (proportion['in_line'] / sum([proportion[key] for key in proportion]))*10 

def is_pronounceable_heuristic(word: str) -> bool:
    #temporary and pretty-good threshold value. Could use some fine-tuning.
    THRESHOLD = 0.35

    # magic list comprehension to extract each sequential pair of letters from the word, and get the pronounceability score for each pair:
    # for example:  
    # "hello" ->  ["he", "el", "ll", "lo"] -> [0.45..., 0.62..., 0.57..., 0.44...]
    average_score = [pronounceable_score_heuristic(pair) for pair in \
        get_bigrams(word)]

    # if the average pronounceability score is too low, we assume it isn't pronounceable.

    return mean(average_score) >= THRESHOLD


# this function allows for more concise and readable code in our __main__ flow.
# uses a contextlib contextmanager to implement __enter__ and __exit__ for our function
# so we can use it in 'with' statements.
@contextmanager
def heuristic_function():
    function = is_pronounceable_heuristic
    try:
        yield function
    finally:
        pass


def is_pronounceable_svm(word: str):
    model = svm.LinearSVC()
    letter_pairs = get_bigrams(word)

@contextmanager
def svm_function():
    function = is_pronounceable_svm
    try:
        yield function
    finally:
        pass


if __name__ == '__main__':
    data = {"hello": True} # index is word, value is whether we deem it pronounceable as training data?


    
    for word in data:
        # test our likelihood heuristic on our data
        with heuristic_function() as is_pronounceable:
            print(f"Heuristic output: {is_pronounceable(word)}")
        # test an SVM
        with svm_function() as is_pronounceable:
            print(f"SVM output: {is_pronounceable(word)}")
        
        # test LDA
        #TODO


