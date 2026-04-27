'''
Statistical sample size bounds for bias estimation under sampling without replacement.

Implements Hoeffding and Serfling-based bounds for computing the minimum number of
samples needed to estimate group-label frequencies to within epsilon error with
(1 - delta) confidence. Used in Section 4.4 of the paper to determine coverage
thresholds for external source distribution estimation.

Reference: Scarone et al., 'Data Bias Mitigation under Coverage Constraints &
The Price of Fairness', FAccT 2026.
'''
import pandas as pd
import math
import numpy as np
import pprint
from dataframe_sketches import (get_total_count, add_dicts)

def sample_size_Hoeffding(k:int,eps:float,delta:float)->int:
    '''
        k is the number of group-label pairs
    '''
    # calculates the natural logarithm by default
    return round(math.log(2*k/delta)/(2*eps**2)) 

def sample_size_Serfling(N:int,k:int,eps:float,delta:float)->int:
    '''
        k is the number of group-label pairs
    '''
    # calculates the natural logarithm by default
    L:float = math.log(2*k/delta)
    nominator:float = (N+1)*L
    denominator:float = L+2*eps**2*N
    return round(nominator/denominator) 

def sample_tuples(group_counts:dict,n:int,replace:bool):
    '''
        Sample n tuples uniformly from a table with given group counts.
        # Example usage
        group_counts = {"group_A": 100, "group_B": 50, "group_C": 30}
        samples = sample_tuples(group_counts, 20, True)    
    '''
    # Create array of group labels repeated by their counts
    groups = np.repeat(list(group_counts.keys()), list(group_counts.values()))
    
    # Sample uniformly
    return np.random.choice(groups, size=n, replace=replace)

def compute_approx_frequencies(n:int,group_counts:dict,replace:bool)->dict:
    '''
        n is the number of samples to be used
    '''
    # sample n tuples from the group_counts distribution
    sample = sample_tuples(group_counts,n,replace)
    # compute approx counts and frequencies from the sample \hat{p}_i
    sample_counts = {gl:list(sample).count(gl) for gl in group_counts.keys()}
    # print(f'sample_counts={sample_counts}')
    # print(f'total samples = {sum(list(sample_counts.values()))}')
    sample_freqs = {gl:sample_counts[gl]/n for gl in sample_counts.keys()}
    # print(sample_freqs)
    return sample_freqs

def compute_sample_error(true_distribution:dict,approx_distribution:dict)->dict:
    # compute error |p_i-\hat{p}_i|
    grouplabel_error = {gl:abs(true_distribution[gl]-approx_distribution[gl]) 
                        for gl in true_distribution.keys()}
    return grouplabel_error

if __name__ == "__main__":
    sketch = pd.read_csv(
        'data/compas/compas_Sex_Code_Text_race_binary_ScoreText_sketch.csv'
    )

    N = get_total_count(sketch)
    sketch['freqs'] = sketch['count'] / N
    sketch['grouplabel'] = (
        sketch['Sex_Code_Text'].astype(str)
        + sketch['race_binary'].astype(str)
        + sketch['ScoreText'].astype(str)
    )

    grouplabel_freqs = list(sketch['freqs'])
    grouplabel_names = list(sketch['grouplabel'])
    true_freqs = dict(zip(grouplabel_names, grouplabel_freqs))
    true_counts = dict(zip(grouplabel_names, list(sketch['count'])))

    k = len(grouplabel_names)  # 12: 4 groups x 3 labels
    eps = 0.05
    delta = 0.05

    n_H = sample_size_Hoeffding(k, eps, delta)
    n_S = sample_size_Serfling(N, k, eps, delta)
    print(f'N={N}, n_H={n_H}, n_S={n_S}')

    repetitions = 100
    avg_error_H = {gl: 0 for gl in grouplabel_names}
    avg_error_S = {gl: 0 for gl in grouplabel_names}
    for _ in range(repetitions):
        sample_freqs_H = compute_approx_frequencies(n_H, true_counts, replace=False)
        sample_freqs_S = compute_approx_frequencies(n_S, true_counts, replace=False)
        avg_error_H = add_dicts(avg_error_H, compute_sample_error(true_freqs, sample_freqs_H))
        avg_error_S = add_dicts(avg_error_S, compute_sample_error(true_freqs, sample_freqs_S))

    avg_error_H = {gl: avg_error_H[gl] / repetitions for gl in avg_error_H}
    avg_error_S = {gl: avg_error_S[gl] / repetitions for gl in avg_error_S}

    avg_error_per_H = {gl: round(avg_error_H[gl] * 100, 2) for gl in avg_error_H}
    avg_error_per_S = {gl: round(avg_error_S[gl] * 100, 2) for gl in avg_error_S}
    pprint.pprint({'Hoeffding errors (%)': avg_error_per_H,
                   'Serfling errors (%)': avg_error_per_S})