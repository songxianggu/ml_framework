from typing import List
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def calculate_chi_square(feature : [], label : []) -> float:
    threshold = np.mean(feature)
    # Binarize feature: 1 if greater than mean, else 0
    feature_binary = [x > threshold for x in feature]

    assert set(np.unique(label)) <= {0, 1}, "label must be binary"
    assert set(np.unique(feature_binary)) <= {0, 1}, "feature must be binary"

    n = len(feature_binary)
    A = np.array(feature_binary)
    B = np.array(label)

    # Probabilities
    pA  = A.mean()
    pNotA = 1 - pA
    pB  = B.mean()
    pNotB = 1 - pB

    pAB      = ((A == 1) & (B == 1)).sum() / n
    pA_notB  = ((A == 1) & (B == 0)).sum() / n
    pNotA_B  = ((A == 0) & (B == 1)).sum() / n
    pNotA_NotB = ((A == 0) & (B == 0)).sum() / n

    numerator = (pAB * pNotA_NotB - pA_notB * pNotA_B) ** 2
    denominator = pA * pNotA * pB * pNotB + 1e-12  # avoid divide by zero

    return numerator / denominator


def get_label_counts(label : []) -> []:
    count_0 = sum(1 for x in label if x == 0)
    count_1 = sum(1 for x in label if x == 1)
    return [count_0, count_1]