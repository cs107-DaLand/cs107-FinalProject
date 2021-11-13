import math


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
        if not math.isclose(list1[i], list2[i], rel_tol=1e-9, abs_tol=0.0):
            ans = False
    return ans
