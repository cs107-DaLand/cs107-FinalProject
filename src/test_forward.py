import pytest
import numpy as np
from salad import *

## For comparing derivative dictionaries with rounding
def compare_dicts(dict1, dict2, round_place = 4):
    for k in dict2:
        if np.round(dict1[k], round_place) != np.round(dict2[k], round_place):
            return False
    return True


def test_add_radd():
    x = Variable(3)
    y = x + 3

    assert y.val == 6
    assert list(y.der.values()) == np.array([1])

    x = Variable(3)
    y = 3 + x

    assert y.val == 6
    assert list(y.der.values()) == np.array([1])

    x = Variable(3, {'x': 1})
    y = Variable(3, {'y': 1})
    z = x + y
    assert z.val == 6
    assert z.der == {'x': 1, 'y': 1}


    x = Variable(np.ones((5,5)), label='x')
    y = Variable(np.ones((5,5)), label='y')
    z = x + y
    assert np.array_equal(z.val, 2*np.ones((5,5)))
    np.testing.assert_equal(z.der, {'x': np.ones((5,5)), 'y': np.ones((5,5))})

    z = x+x+y+y+2
    assert np.array_equal(z.val, 4*np.ones((5,5))+2)
    np.testing.assert_equal(z.der, {'x': 2*np.ones((5,5)), 'y': 2*np.ones((5,5))})



def test_sub_rsub():
    x = Variable(3)
    y = x - 3

    assert y.val == 0
    assert list(y.der.values()) == np.array([1])

    x = Variable(3)
    y = 3 - x

    assert y.val == 0
    assert list(y.der.values()) == np.array([-1])

    x = Variable(3, {'x': 1})
    y = Variable(3, {'y': 1})
    z = x - y
    assert z.val == 0
    assert z.der == {'x': 1, 'y': -1}


    x = Variable(np.ones((5,5)), label='x')
    y = Variable(np.ones((5,5)), label='y')
    z = x - y
    assert np.array_equal(z.val, np.zeros((5,5)))
    np.testing.assert_equal(z.der, {'x': np.ones((5,5)), 'y': -1*np.ones((5,5))})

    z = x+x-y-y+2
    assert np.array_equal(z.val, 2*np.ones((5,5)))
    np.testing.assert_equal(z.der, {'x': 2*np.ones((5,5)), 'y': -2*np.ones((5,5))})

def test_mul_rmul():
    x = Variable(3, label='x')
    y = x * 2
    assert y.val == 6
    assert y.der == {'x':  2}

    #y = 5x + x^2
    y = x*2 + 3*x + x * x 
    assert y.val == 24
    assert y.der == {'x': 11}

    x = Variable(3, label='x')
    y = Variable(2, label='y')
    z = x*y
    assert z.val == 6
    assert z.der == {'x': 2, 'y': 3}

    z = 3*z*3
    assert z.val == 54
    assert z.der == {'x': 18, 'y': 27}

    x = Variable(3, label='x')
    y = Variable(2, label='y')
    z = x*y
    z = y*z # y^2*x
    assert z.val == 12
    assert z.der == {'x': y.val**2, 'y': 2*y.val*x.val}


    x = Variable(2*np.ones((5,5)), label='x')
    y = Variable(3*np.ones((5,5)), label='y')
    z = x * y
    assert np.array_equal(z.val, 2*3*np.ones((5,5)))
    np.testing.assert_equal(z.der, {'x': 3*np.ones((5,5)), 'y': 2*np.ones((5,5))})

    z = -1*z * x # f = -(x^2) * y, dx = -2xy, dy = -x^2
    assert np.array_equal(z.val, -12*np.ones((5,5)))
    np.testing.assert_equal(z.der, {'x': -2*2*3*np.ones((5,5)), 'y': -1*2*2*np.ones((5,5))})


def test_truediv_rtruediv():
    x = Variable(3, label='x')
    y = x / 2
    assert y.val == 1.5
    assert y.der == {'x':  1/2}

    
    y = x/2 + 3/x + x / x 
    assert y.val == 3.5
    assert y.der == {'x': .5 - 3/9}

    x = Variable(3, label='x')
    y = Variable(2, label='y')
    z = x/y
    assert z.val == 3/2
    assert z.der == {'x': 1/2, 'y': -3/4} # dx = 1/y, dy = -x/y^2

    z = 2.4/z/x/8 # 2.4/(x/y)/x/8
    assert z.val == 2.4/(3/2)/3/8
    ## Using this function because of rounding errors
    assert compare_dicts(z.der, {'x': (-.6*y.val)/(x.val**3), 'y': (.3/(x.val**2))}) #dx = -.6y/x^3 , dy = .3/x^2
    

    x = Variable(2*np.ones((5,5)), label='x')
    y = Variable(3*np.ones((5,5)), label='y')
    z = x / y
    assert np.array_equal(z.val, 2/3*np.ones((5,5)))
    np.testing.assert_equal(z.der, {'x': 1/y.val, 'y': -1*x.val/(y.val**2)})

    z = -1/ z / x 
    assert np.array_equal(z.val, -1 / (2/3) / 2 *np.ones((5,5)))
    np.testing.assert_equal(z.der, {'x': 2 * y.val / (x.val**3), 'y': -1 / (x.val**2)})
    


test_add_radd()

test_sub_rsub()

test_mul_rmul()

test_truediv_rtruediv()