import math
import numpy as np
import autograd.numpy as adnp
from autograd import grad
import salad as ad
import pytest

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

@pytest.mark.skip
def test_trig(adfunc, adnpfunc):
    '''
    Master test suite for trig functions
    '''
    x = 0.3
    ans = adfunc(x)
    sol = adnpfunc(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = adfunc(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnpfunc(0.3), grad(adnpfunc)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = adfunc(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnpfunc(0.6), grad(lambda x: adnpfunc(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = adfunc(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = adnpfunc(0.7), [
        grad(lambda x, y: adnpfunc(x + y), 0)(0.3, 0.4),
        grad(lambda x, y: adnpfunc(x + y), 1)(0.3, 0.4),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = adfunc(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [adnpfunc(0.3), adnpfunc(0.4), adnpfunc(0.5)], [
        grad(lambda x: adnpfunc(x))(0.3),
        grad(lambda x: adnpfunc(x))(0.4),
        grad(lambda x: adnpfunc(x))(0.5),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = adfunc(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [adnpfunc(0.3 * 2), adnpfunc(0.4 * 2), adnpfunc(0.5 * 2)], [
        grad(lambda x: adnpfunc(x + x))(0.3),
        grad(lambda x: adnpfunc(x + x))(0.4),
        grad(lambda x: adnpfunc(x + x))(0.5),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = adfunc(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnpfunc(0.09), adnpfunc(0.10), adnpfunc(0.11)],
        [
            grad(lambda x, y: adnpfunc(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnpfunc(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnpfunc(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnpfunc(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnpfunc(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnpfunc(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)