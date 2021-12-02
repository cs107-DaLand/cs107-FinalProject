import math
import numpy as np


def add_dict(dict1, dict2):
    """
    Add two dictionaries together.

    Parameters
    ----------
    dict1 : dict
        First dictionary to add.
    dict2 : dict
        Second dictionary to add.
    
    Returns
    -------
    dict
        The sum of the two dictionaries.
    
    Examples
    --------
    >>> add_dict({1: 2, 3: 4}, {1: 3, 5: 6})
    {1: 5, 3: 4, 5: 6}
    """
    # Need to copy dictionaries to prevent changing der for original variables
    dict1 = dict(dict1)
    dict2 = dict(dict2)
    for key in dict2:
        if key in dict1:
            dict2[key] = dict2[key] + dict1[key]
    return {**dict1, **dict2}


def check_list(list1, list2):
    """
    Check if two lists are equal.

    Parameters
    ----------
    list1 : list
        First list to check.
    list2 : list
        Second list to check.
    
    Returns
    -------
    bool
        True if the lists are equal, False otherwise.
    
    Examples
    --------
    >>> check_list([1, 2, 3], [1, 2, 3])
    True
    >>> check_list([1, 2, 3], [1, 2, 4])
    False
    """
    assert len(list1) == len(list2)
    ans = True

    for i in range(len(list1)):
        if not math.isclose(list1[i], list2[i], rel_tol=1e-9, abs_tol=0.0):
            ans = False
    return ans

def compare_dicts(dict1, dict2, round_place=4):
    """
    Check if two dictionaries are equal (with rounding)
    
    Parameters
    ----------
    dict1 : dict
        First dictionary to check.
    dict2 : dict
        Second dictionary to check.
    round_place : int
        Number of decimal places to round to.
    
    Returns
    -------
    bool
        True if the dictionaries are equal, False otherwise.
    
    Examples
    --------
    >>> compare_dicts({'x': 1.0000, 'y':1.1234}, {'x': 1.0000, 'y':1.1234})
    True
    >>> compare_dicts({'x': 1.0000, 'y':1.1234}, {'x': 1.0000, 'y':1.1235})
    False
    """
    for k in dict2:
        if np.round(dict1[k], round_place) != np.round(dict2[k], round_place):
            return False
    return True

def compare_dicts_multi(dict1, dict2):
    """
    Check if two dictionaries are equal

    Parameters
    ----------
    dict1 : dict
        First dictionary to check.
    dict2 : dict
        Second dictionary to check.
    
    Returns
    -------
    bool
        True if the dictionaries are equal, False otherwise.
    
    Examples
    --------
    >>> compare_dicts_multi({'x': 1.0000, 'y':1.1234}, {'x': 1.0000, 'y':1.1234})
    True
    >>> compare_dicts_multi({'x': 1.0000, 'y':1.12324}, {'x': 1.0000, 'y':1.12325})
    False
    """
    if not set(dict1) == set(dict2):
        return False
    for k in dict1:
        if not np.all(dict1[k] == dict2[k]):
            return False
    return True