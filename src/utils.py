import math
import numpy as np

def add_dict(dict1, dict2):
    # Need to copy dictionaries to prevent changing der for original variables
    dict1 = dict(dict1)
    dict2 = dict(dict2)
    for key in dict2:
        if key in dict1:
            dict2[key] = dict2[key] + dict1[key]
    return {**dict1, **dict2}


def check_list(list1, list2):
    assert len(list1) == len(list2)
    ans = True

    for i in range(len(list1)):
        if not math.isclose(list1[i], list2[i], rel_tol=1e-7, abs_tol=0.0):
            print(f'Check_list violated: {list1[i]} != {list2[i]}')
            ans = False
    return ans

## For comparing derivative dictionaries with rounding
def compare_dicts(dict1, dict2, round_place=4):
    for k in dict2:
        if np.round(dict1[k], round_place) != np.round(dict2[k], round_place):
            return False
    return True

def compare_dicts_multi(d1, d2):
    if not set(d1) == set(d2):
        return False
    for k in d1:
        if not np.all(d1[k] == d2[k]):
            return False
    return True
    