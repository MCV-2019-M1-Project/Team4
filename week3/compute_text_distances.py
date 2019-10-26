import textdistance

def levenshtein_distance(str_1, str_2):
    """
        The Levenshtein distance is a string metric for measuring the difference between two sequences.
        It is calculated as the minimum number of single-character edits necessary to transform one string into another.
    """
    return textdistance.levenshtein.normalized_similarity(str_1, str_2)


def hamming_distance(str_1, str_2):
    """
        The Hamming distance is a string metric for measuring the difference between two sequences.
        It is calculated as the finding the number of places where the strings vary.
    """
    return textdistance.hamming.normalized_similarity(str_1, str_2)

def jaro_winkler_distance(str_1, str_2):
    """
        The Levenshtein distance is a string metric for measuring the difference between two sequences.
        This algorithms gives high scores to two strings if, (1) they contain same characters, but within a certain distance from one another, and (2) the order of the matching characters is same.
    """
    return textdistance.jaro_winkler(str_1, str_2)



