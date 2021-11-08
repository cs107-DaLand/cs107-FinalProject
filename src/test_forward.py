import pytest
import numpy as np
import salad as ad

## For comparing derivative dictionaries with rounding
def compare_dicts(dict1, dict2, round_place=4):
    for k in dict2:
        if np.round(dict1[k], round_place) != np.round(dict2[k], round_place):
            return False
    return True


def test_neg():
    print("----------Testing neg-----------")
    print("***x=3, y=-x***")
    x = ad.Variable(3, label='x')
    y = -x
    print(y)

    print("***x=3, x.der={'x':2}, y=-x***")
    x = ad.Variable(3, label='x', der={'x':2})
    y = -x
    print(y)

    print("***x=[0,1,2], y=-x***")
    x = ad.Variable(np.arange(3), label='x')
    y = -x
    print(y)

    print("***x=0, y=3, z=x+2y, z2=-z***")
    x = ad.Variable(0, label='x')
    y = ad.Variable(3, label='y')
    z = x + 2 * y
    z2 = -z
    print(z2)

    print("***x=[0,1,2], y=[3,4,5], z=x+2y, z2=-z***")
    x = ad.Variable(np.arange(3), label='x')
    y = ad.Variable(3 + np.arange(3), label='y')
    z = x + 2 * y
    z2 = -z
    print(z2)

def test_pow():
    print("----------Testing pow-----------")
    print("***x=3, z=x^2***")
    print("should be: z=9, dz/dx=6")
    x = ad.Variable(3, label='x')
    z = x ** 2
    print(z)

    print("***x=[3,2], z=x^2***")
    print("should be: z=[9,4], dz/dx=[6,4]")
    x = ad.Variable([3,2], label='x')
    z = x ** 2
    print(z)

    print("***x=3, y=2, z=x^y***")
    print("should be: z=9, dz/dx=6, dz/dy=9.8875")
    x = ad.Variable(3, label='x')
    y = ad.Variable(2, label='y')
    z = x ** y
    print(z)

    print("***x=[3,2], y=[2,3], z=x^y***")
    print("should be: z=[9,8], dz/dx=[6,12], dz/dy=[9.8875,5.5452]")
    x = ad.Variable([3,2], label='x')
    y = ad.Variable([2,3], label='y')
    z = x ** y
    print(z)

    print("***x=[e-1, e-1], y=[1,1], z=x+y, z2=z^y***")
    print("should be: z2=[e,e], dz/dx=[1,1], dz/dy=[e+1, e+1]")
    x = ad.Variable([np.e-1, np.e-1], label='x')
    y = ad.Variable([1,1], label='y')
    z = x + y
    z2 = z ** y
    print(z2)

def test_rpow():
    print("----------Testing rpow-----------")
    print("***x=1, z=e^x***")
    print("should be: z=e, dz/dx=e")
    x = ad.Variable(1, label='x')
    z = np.e ** x
    print(z)

    print("***x=[1,2], z=e^x***")
    print("should be: z=[e, 7.389], dz/dx=[e, 7.389]")
    x = ad.Variable([1,2], label='x')
    z = np.e ** x
    print(z)

    print("***x=2, y=-1, z=e^(x+2*y)***")
    print("should be: z=e, dz/dx=1, dz/dy=2")
    x = ad.Variable(2, label='x')
    y = ad.Variable(-1, label='y')
    z = np.e ** (x + 2 * y)
    print(z)

    print("***x=[2,-2], y=[-1,1], z=e^(x+2*y)***")
    print("should be: z=[e,e], dz/dx=[1,1], dz/dy=[2,2]")
    x = ad.Variable([2,-2], label='x')
    y = ad.Variable([-1,1], label='y')
    z = np.e ** (x + 2 * y)
    print(z)

def test_ne():
    print("----------Testing ne-----------")
    print("!!!!!!!!NOTICE: ONLY COMPARING LABEL!!!!!!!!")
    print("***x=1, y=1***")
    print("should be: x==x, x!=y")
    x = ad.Variable(1, label='x')
    y = ad.Variable(1, label='y')
    print("x!=x: ", x!=x)
    print("x!=y: ", x!=y)

def test_lt():
    print("----------Testing lt-----------")
    print("***x=1, y=2***")
    print("should be: x<y")
    x = ad.Variable(1, label='x')
    y = ad.Variable(2, label='y')
    print("x<y: ", x<y)

    print("***x=[1,2], y=[2,2]***")
    print("should be: [T, F]")
    x = ad.Variable([1,2], label='x')
    y = ad.Variable([2,2], label='y')
    print("x<y: ", x<y)

def test_le():
    print("----------Testing le-----------")
    print("***x=1, y=2***")
    print("should be: x<=y")
    x = ad.Variable(1, label='x')
    y = ad.Variable(2, label='y')
    print("x<=y: ", x<=y)

    print("***x=[1,2], y=[2,2]***")
    print("should be: [T, T]")
    x = ad.Variable([1,2], label='x')
    y = ad.Variable([2,2], label='y')
    print("x<=y: ", x<=y)

def test_gt():
    print("----------Testing gt-----------")
    print("***x=3, y=2***")
    print("should be: x>y")
    x = ad.Variable(3, label='x')
    y = ad.Variable(2, label='y')
    print("x>y: ", x>y)

    print("***x=[3,2], y=[2,2]***")
    print("should be: [T, F]")
    x = ad.Variable([3,2], label='x')
    y = ad.Variable([2,2], label='y')
    print("x>y: ", x>y)

def test_ge():
    print("----------Testing ge-----------")
    print("***x=3, y=2***")
    print("should be: x>=y")
    x = ad.Variable(3, label='x')
    y = ad.Variable(2, label='y')
    print("x>=y: ", x>=y)

    print("***x=[3,2], y=[2,2]***")
    print("should be: [T, T]")
    x = ad.Variable([3,2], label='x')
    y = ad.Variable([2,2], label='y')
    print("x>=y: ", x>=y)


if __name__ == "__main__":
    test_neg()
    test_pow()
    test_rpow()
    test_ne()
    test_lt()
    test_le()
    test_gt()
    test_ge()