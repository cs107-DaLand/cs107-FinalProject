import pytest
import math
import numpy as np
import autograd.numpy as adnp
from autograd import grad
import salad as ad
from utils import check_list, compare_dicts, compare_dicts_multi


def test_add_radd():
    x = ad.Variable(3)
    y = x + 3

    assert y.val == 6
    assert list(y.der.values()) == np.array([1])

    x = ad.Variable(3)
    y = 3 + x

    assert y.val == 6
    assert list(y.der.values()) == np.array([1])

    x = ad.Variable(3, {"x": 1})
    y = ad.Variable(3, {"y": 1})
    z = x + y
    assert z.val == 6
    assert z.der == {"x": 1, "y": 1}

    x = ad.Variable(np.ones((5, 5)), label="x")
    y = ad.Variable(np.ones((5, 5)), label="y")
    z = x + y
    assert np.array_equal(z.val, 2 * np.ones((5, 5)))
    np.testing.assert_equal(z.der, {"x": np.ones((5, 5)), "y": np.ones((5, 5))})

    z = x + x + y + y + 2
    assert np.array_equal(z.val, 4 * np.ones((5, 5)) + 2)
    np.testing.assert_equal(z.der, {"x": 2 * np.ones((5, 5)), "y": 2 * np.ones((5, 5))})


def test_sub_rsub():
    x = ad.Variable(3)
    y = x - 3

    assert y.val == 0
    assert list(y.der.values()) == np.array([1])

    x = ad.Variable(3)
    y = 3 - x

    assert y.val == 0
    assert list(y.der.values()) == np.array([-1])

    x = ad.Variable(3, {"x": 1})
    y = ad.Variable(3, {"y": 1})
    z = x - y
    assert z.val == 0
    assert z.der == {"x": 1, "y": -1}

    x = ad.Variable(np.ones((5, 5)), label="x")
    y = ad.Variable(np.ones((5, 5)), label="y")
    z = x - y
    assert np.array_equal(z.val, np.zeros((5, 5)))
    np.testing.assert_equal(z.der, {"x": np.ones((5, 5)), "y": -1 * np.ones((5, 5))})

    z = x + x - y - y + 2
    assert np.array_equal(z.val, 2 * np.ones((5, 5)))
    np.testing.assert_equal(
        z.der, {"x": 2 * np.ones((5, 5)), "y": -2 * np.ones((5, 5))}
    )


def test_mul_rmul():
    x = ad.Variable(3, label="x")
    y = x * 2
    assert y.val == 6
    assert y.der == {"x": 2}

    # y = 5x + x^2
    y = x * 2 + 3 * x + x * x
    assert y.val == 24
    assert y.der == {"x": 11}

    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    z = x * y
    assert z.val == 6
    assert z.der == {"x": 2, "y": 3}

    z = 3 * z * 3
    assert z.val == 54
    assert z.der == {"x": 18, "y": 27}

    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    z = x * y
    z = y * z  # y^2*x
    assert z.val == 12
    assert z.der == {"x": y.val ** 2, "y": 2 * y.val * x.val}

    x = ad.Variable(2 * np.ones((5, 5)), label="x")
    y = ad.Variable(3 * np.ones((5, 5)), label="y")
    z = x * y
    assert np.array_equal(z.val, 2 * 3 * np.ones((5, 5)))
    np.testing.assert_equal(z.der, {"x": 3 * np.ones((5, 5)), "y": 2 * np.ones((5, 5))})

    z = -1 * z * x  # f = -(x^2) * y, dx = -2xy, dy = -x^2
    assert np.array_equal(z.val, -12 * np.ones((5, 5)))
    np.testing.assert_equal(
        z.der, {"x": -2 * 2 * 3 * np.ones((5, 5)), "y": -1 * 2 * 2 * np.ones((5, 5))}
    )


def test_truediv_rtruediv():
    x = ad.Variable(3, label="x")
    y = x / 2
    assert y.val == 1.5
    assert y.der == {"x": 1 / 2}

    y = x / 2 + 3 / x + x / x
    assert y.val == 3.5
    assert y.der == {"x": 0.5 - 3 / 9}

    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    z = x / y
    assert z.val == 3 / 2
    assert z.der == {"x": 1 / 2, "y": -3 / 4}  # dx = 1/y, dy = -x/y^2

    z = 2.4 / z / x / 8  # 2.4/(x/y)/x/8
    assert z.val == 2.4 / (3 / 2) / 3 / 8
    ## Using this function because of rounding errors
    assert compare_dicts(
        z.der, {"x": (-0.6 * y.val) / (x.val ** 3), "y": (0.3 / (x.val ** 2))}
    )  # dx = -.6y/x^3 , dy = .3/x^2

    x = ad.Variable(2 * np.ones((5, 5)), label="x")
    y = ad.Variable(3 * np.ones((5, 5)), label="y")
    z = x / y
    assert np.array_equal(z.val, 2 / 3 * np.ones((5, 5)))
    np.testing.assert_equal(z.der, {"x": 1 / y.val, "y": -1 * x.val / (y.val ** 2)})

    z = -1 / z / x
    assert np.array_equal(z.val, -1 / (2 / 3) / 2 * np.ones((5, 5)))
    np.testing.assert_equal(
        z.der, {"x": 2 * y.val / (x.val ** 3), "y": -1 / (x.val ** 2)}
    )


def test_exp():
    x = 3
    ans = ad.exp(x)
    sol = np.exp(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = np.exp(3), np.exp(3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x")
    ans_val, ans_der = ad.exp(x).val, ad.exp(x).der["x"]
    sol_val, sol_der = np.exp(3), np.exp(3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = np.exp(6), np.exp(6)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.exp(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = np.exp(7), [np.exp(7), np.exp(7)]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [np.exp(3), np.exp(4), np.exp(5)], [
        np.exp(3),
        np.exp(4),
        np.exp(5),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.exp(x) + ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [2 * np.exp(3), 2 * np.exp(4), 2 * np.exp(5)], [
        2 * np.exp(3),
        2 * np.exp(4),
        2 * np.exp(5),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    z = x + x
    y = ad.exp(z)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [np.exp(2 * 3), np.exp(2 * 4), np.exp(2 * 5)], [
        2 * np.exp(2 * 3),
        2 * np.exp(2 * 4),
        2 * np.exp(2 * 5),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.exp(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [np.exp(9), np.exp(10), np.exp(11)],
        [
            grad(lambda x, y: adnp.exp(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: adnp.exp(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: adnp.exp(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: adnp.exp(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: adnp.exp(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: adnp.exp(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_ln():
    x = 3
    ans = ad.ln(x)
    sol = adnp.log(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.ln(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.log(3), grad(adnp.log)(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.ln(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.log(6), grad(lambda x: adnp.log(x + 3.0))(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.ln(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = adnp.log(7), [
        grad(lambda x, y: adnp.log(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: adnp.log(x + y), 1)(3.0, 4.0),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.ln(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [np.log(3), np.log(4), np.log(5)], [
        grad(lambda x: adnp.log(x))(3.0),
        grad(lambda x: adnp.log(x))(4.0),
        grad(lambda x: adnp.log(x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.ln(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [adnp.log(3 * 2), adnp.log(4 * 2), adnp.log(5 * 2)], [
        grad(lambda x: adnp.log(x + x))(3.0),
        grad(lambda x: adnp.log(x + x))(4.0),
        grad(lambda x: adnp.log(x + x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.ln(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [np.log(9), np.log(10), np.log(11)],
        [
            grad(lambda x, y: adnp.log(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: adnp.log(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: adnp.log(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: adnp.log(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: adnp.log(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: adnp.log(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_logistic():
    def logistic(x):
        return 1 / (1 + adnp.exp(-x))

    x = 3
    ans = ad.logistic(x)
    sol = logistic(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.logistic(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = logistic(3), grad(logistic)(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.logistic(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = logistic(6), grad(lambda x: logistic(x + 3.0))(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.logistic(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = logistic(7), [
        grad(lambda x, y: logistic(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: logistic(x + y), 1)(3.0, 4.0),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.logistic(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [logistic(3), logistic(4), logistic(5)], [
        grad(lambda x: logistic(x))(3.0),
        grad(lambda x: logistic(x))(4.0),
        grad(lambda x: logistic(x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.logistic(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [logistic(3 * 2), logistic(4 * 2), logistic(5 * 2)], [
        grad(lambda x: logistic(x + x))(3.0),
        grad(lambda x: logistic(x + x))(4.0),
        grad(lambda x: logistic(x + x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.logistic(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [logistic(9), logistic(10), logistic(11)],
        [
            grad(lambda x, y: logistic(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: logistic(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: logistic(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: logistic(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: logistic(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: logistic(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_log10():
    def log10(x):
        return adnp.log(x) / adnp.log(10)

    x = 3
    ans = ad.log10(x)
    sol = log10(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.log10(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = log10(3), grad(log10)(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.log10(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = log10(6), grad(lambda x: log10(x + 3.0))(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.log10(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = log10(7), [
        grad(lambda x, y: log10(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: log10(x + y), 1)(3.0, 4.0),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.log10(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [log10(3), log10(4), log10(5)], [
        grad(lambda x: log10(x))(3.0),
        grad(lambda x: log10(x))(4.0),
        grad(lambda x: log10(x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.log10(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [log10(3 * 2), log10(4 * 2), log10(5 * 2)], [
        grad(lambda x: log10(x + x))(3.0),
        grad(lambda x: log10(x + x))(4.0),
        grad(lambda x: log10(x + x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.log10(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [log10(9), log10(10), log10(11)],
        [
            grad(lambda x, y: log10(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: log10(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: log10(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: log10(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: log10(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: log10(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_sin():
    def sin(x):
        return adnp.sin(x)

    x = 3
    ans = ad.sin(x)
    sol = sin(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.sin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = sin(3), grad(sin)(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.sin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = sin(6), grad(lambda x: sin(x + 3.0))(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.sin(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = sin(7), [
        grad(lambda x, y: sin(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: sin(x + y), 1)(3.0, 4.0),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.sin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [sin(3), sin(4), sin(5)], [
        grad(lambda x: sin(x))(3.0),
        grad(lambda x: sin(x))(4.0),
        grad(lambda x: sin(x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.sin(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [sin(3 * 2), sin(4 * 2), sin(5 * 2)], [
        grad(lambda x: sin(x + x))(3.0),
        grad(lambda x: sin(x + x))(4.0),
        grad(lambda x: sin(x + x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.sin(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [sin(9), sin(10), sin(11)],
        [
            grad(lambda x, y: sin(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: sin(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: sin(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: sin(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: sin(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: sin(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_cos():
    def cos(x):
        return adnp.cos(x)

    x = 3
    ans = ad.cos(x)
    sol = cos(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.cos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = cos(3), grad(cos)(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.cos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = cos(6), grad(lambda x: cos(x + 3.0))(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.cos(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = cos(7), [
        grad(lambda x, y: cos(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: cos(x + y), 1)(3.0, 4.0),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.cos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [cos(3), cos(4), cos(5)], [
        grad(lambda x: cos(x))(3.0),
        grad(lambda x: cos(x))(4.0),
        grad(lambda x: cos(x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.cos(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [cos(3 * 2), cos(4 * 2), cos(5 * 2)], [
        grad(lambda x: cos(x + x))(3.0),
        grad(lambda x: cos(x + x))(4.0),
        grad(lambda x: cos(x + x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.cos(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [cos(9), cos(10), cos(11)],
        [
            grad(lambda x, y: cos(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: cos(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: cos(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: cos(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: cos(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: cos(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_tan():
    def tan(x):
        return adnp.tan(x)

    x = 3
    ans = ad.tan(x)
    sol = tan(x)
    assert sol == ans

    x = ad.Variable(3, label="x")
    y = ad.tan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = tan(3), grad(tan)(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + 3
    y = ad.tan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = tan(6), grad(lambda x: tan(x + 3.0))(3.0)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.tan(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = tan(7), [
        grad(lambda x, y: tan(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: tan(x + y), 1)(3.0, 4.0),
    ]
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.tan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [tan(3), tan(4), tan(5)], [
        grad(lambda x: tan(x))(3.0),
        grad(lambda x: tan(x))(4.0),
        grad(lambda x: tan(x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.tan(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = [tan(3 * 2), tan(4 * 2), tan(5 * 2)], [
        grad(lambda x: tan(x + x))(3.0),
        grad(lambda x: tan(x + x))(4.0),
        grad(lambda x: tan(x + x))(5.0),
    ]
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.Variable([6, 6, 6], label="y")
    y = ad.tan(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [tan(9), tan(10), tan(11)],
        [
            grad(lambda x, y: tan(x + y), 0)(3.0, 6.0),
            grad(lambda x, y: tan(x + y), 0)(4.0, 6.0),
            grad(lambda x, y: tan(x + y), 0)(5.0, 6.0),
        ],
        [
            grad(lambda x, y: tan(x + y), 1)(3.0, 6.0),
            grad(lambda x, y: tan(x + y), 1)(4.0, 6.0),
            grad(lambda x, y: tan(x + y), 1)(5.0, 6.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_neg():

    x = ad.Variable(3, label="x")
    y = -x
    assert y.val == -3
    assert y.der == {"x": -1}

    x = ad.Variable(3, label="x", der={"x": 2})
    y = -x
    assert y.val == -3
    assert y.der == {"x": -2}

    x = ad.Variable(np.arange(3), label="x")
    y = -x
    assert np.all(y.val == [0, -1, -2])
    assert y.der == {"x": [-1, -1, -1]}

    x = ad.Variable(0, label="x")
    y = ad.Variable(3, label="y")
    z = x + 2 * y
    z2 = -z
    assert z2.val == -6
    assert z2.der == {"x": -1, "y": -2}

    x = ad.Variable(np.arange(3), label="x")
    y = ad.Variable(3 + np.arange(3), label="y")
    z = x + 2 * y
    z2 = -z
    assert np.all(z2.val == [-6, -9, -12])
    assert z2.der == {"x": [-1, -1, -1], "y": [-2, -2, -2]}


def test_pow():
    x = ad.Variable(3, label="x")
    z = x ** 2
    assert z.val == 9
    assert z.der == {"x": 6}

    x = ad.Variable([3, 2], label="x")
    z = x ** 2
    assert np.all(z.val == [9, 4])
    assert np.all(z.der == {"x": [6, 4]})

    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    z = x ** y
    assert z.val == 9
    assert z.der == {"x": 6, "y": 9 * np.log(3)}

    x = ad.Variable([3, 2], label="x")
    y = ad.Variable([2, 3], label="y")
    z = x ** y
    assert np.all(z.val == [9, 8])
    assert (
        compare_dicts_multi(z.der, {"x": [6, 12], "y": [9 * np.log(3), 8 * np.log(2)]})
        == True
    )

    x = ad.Variable([np.e - 1, np.e - 1], label="x")
    y = ad.Variable([1, 1], label="y")
    z = x + y
    z2 = z ** y
    assert np.all(z2.val == [np.e, np.e])
    assert compare_dicts_multi(z2.der, {"x": [1, 1], "y": [np.e + 1, np.e + 1]}) == True


def test_rpow():
    x = ad.Variable(1, label="x")
    z = np.e ** x
    assert z.val == np.e
    assert z.der == {"x": np.e}

    x = ad.Variable([1, 2], label="x")
    z = np.e ** x
    assert np.all(z.val == [np.e, np.e ** 2])
    assert np.all(z.der == {"x": [np.e, np.e ** 2]})

    x = ad.Variable(2, label="x")
    y = ad.Variable(-1, label="y")
    z = np.e ** (x + 2 * y)
    assert z.val == 1
    assert z.der == {"x": 1, "y": 2}

    x = ad.Variable([2, -2], label="x")
    y = ad.Variable([-1, 1], label="y")
    z = np.e ** (x + 2 * y)
    assert np.all(z.val == [1, 1])
    assert np.all(z.der == {"x": [1, 1], "y": [2, 2]})


def test_ne():
    x = ad.Variable(1, label="x")
    y = ad.Variable(1, label="y")
    assert (x != x) == False
    assert (x != y) == True

    z1 = ad.Variable([1, 2], der={"x": [1, 2], "y": [1, 2]}, label="z1")
    z2 = ad.Variable([1, 2], der={"x": [1, 2], "y": [1, 2]}, label="z2")
    assert (z1 != z2) == False

    z1 = ad.Variable(1, der={"x": 2, "y": 3}, label="z1")
    z2 = ad.Variable(1, der={"x": 2, "y": 3}, label="z2")
    assert (z1 != z2) == False

    z1 = ad.Variable([1, 2], der={"x": [1, 2], "y": [1, 2]}, label="z1")
    z2 = ad.Variable([1, 2], der={"x": [1, 2], "y": [1, 3]}, label="z2")
    assert (z1 != z2) == True

    x = ad.Variable(1, label="x")
    y = ad.Variable(1, label="y")
    z1 = ad.exp(x) + np.e * y
    z2 = ad.exp(y) + np.e * x
    assert (z1 != z2) == False


def test_lt():
    x = ad.Variable(1, label="x")
    y = ad.Variable(2, label="y")
    assert (x < y) == True

    x = ad.Variable([1, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x < y) == [True, False])


def test_le():
    x = ad.Variable(1, label="x")
    y = ad.Variable(2, label="y")
    assert (x <= y) == True

    x = ad.Variable([1, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x <= y) == [True, True])


def test_gt():
    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    assert (x > y) == True

    x = ad.Variable([3, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x > y) == [True, False])


def test_ge():
    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    assert (x >= y) == True

    x = ad.Variable([3, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x >= y) == [True, True])


def test_complicated_functions():
    ## Function 1
    ## sin(x) + cos(x) * 3*y - x^4 + ln(x*y)
    x = np.random.rand(5, 4)
    x_var = ad.Variable(x, label="x")
    y = np.random.rand(5, 4)
    y_var = ad.Variable(y, label="y")

    f_ad = ad.sin(x_var) + ad.cos(x_var) * 3 * y_var - x_var ** 4 + ad.ln(x_var * y_var)
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    f_np_val = np.sin(x) + np.cos(x) * 3 * y - x ** 4 + np.log(x * y)
    dx = -4 * x ** 3 - 3 * y * np.sin(x) + 1 / x + np.cos(x)
    dy = 3 * np.cos(x) + 1 / y

    assert np.array_equal(f_ad_val, f_np_val)
    assert np.array_equal(np.around(f_ad_grad["x"], 4), np.around(dx, 4))
    assert np.array_equal(np.around(f_ad_grad["y"], 4), np.around(dy, 4))

    ## Function 2
    ## cos(x*y^2) + exp(x*y*3x)
    x = np.random.rand(3, 8)
    x_var = ad.Variable(x, label="x")
    y = np.random.rand(3, 8)
    y_var = ad.Variable(y, label="y")

    f_ad = ad.cos(x_var * y_var ** 2) + ad.exp(x_var * y_var * 3 * x_var)
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    f_np_val = np.cos(x * y ** 2) + np.exp(x * y * 3 * x)
    dx = y * (6 * x * np.exp(3 * x ** 2 * y) - y * np.sin(x * y ** 2))
    dy = x * (3 * x * np.exp(3 * x ** 2 * y) - 2 * y * np.sin(x * y ** 2))

    assert np.array_equal(f_ad_val, f_np_val)
    assert np.array_equal(np.around(f_ad_grad["x"], 4), np.around(dx, 4))
    assert np.array_equal(np.around(f_ad_grad["y"], 4), np.around(dy, 4))

    ## Function 3
    ## tan(x+y/2) - ln(z/x)
    x = np.random.rand(10, 10)
    x_var = ad.Variable(x, label="x")
    y = np.random.rand(10, 10)
    y_var = ad.Variable(y, label="y")

    f_ad = ad.tan(x_var + y_var / 2) - ad.ln(y_var / x_var)
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    f_np_val = np.tan(x + y / 2) - np.log(y / x)
    dx = (1 / np.cos(x + y / 2)) ** 2 + 1 / x
    dy = 1 / 2 * (1 / np.cos(x + y / 2)) ** 2 - 1 / y

    assert np.array_equal(np.around(f_ad_val, 4), np.around(f_np_val, 4))
    assert np.array_equal(np.around(f_ad_grad["x"], 4), np.around(dx, 4))
    assert np.array_equal(np.around(f_ad_grad["y"], 4), np.around(dy, 4))

    ## Function 4
    ## (sin(2*x) + cos(x/2) * y - ln(x^4)) / (x^2 + y^2)

    x = 5
    x_var = ad.Variable(x, label="x")
    y = 7
    y_var = ad.Variable(y, label="y")

    f_ad = (ad.sin(x_var * 2) + ad.cos(x_var / 2) * y_var - ad.ln(x_var ** 4)) / (
        x_var ** 2 + y_var ** 2
    )
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    def f4(x, y):
        return (adnp.sin(x * 2) + adnp.cos(x / 2) * y - adnp.log(x ** 4)) / (
            x ** 2 + y ** 2
        )

    f_np_val, dx, dy = [
        f4(x, y),
        grad(lambda x, y: f4(x, y), 0)(5.0, 7.0),
        grad(lambda x, y: f4(x, y), 1)(5.0, 7.0),
    ]

    assert np.around(f_ad_val, 4) == np.around(f_np_val, 4)
    assert np.around(f_ad_grad["x"], 4) == np.around(dx, 4)
    assert np.around(f_ad_grad["y"], 4) == np.around(dy, 4)

    ## Function 5
    ## (tan(2*x-y/2) * y - exp(x^2 / y^2)

    x = 5
    x_var = ad.Variable(x, label="x")
    y = 7
    y_var = ad.Variable(y, label="y")

    f_ad = ad.tan(2 * x_var - y_var / 2) * y_var - ad.exp(x_var ** 2 / y_var ** 2)
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    def f5(x, y):
        return adnp.tan(2 * x - y / 2) * y - adnp.exp((x ** 2) / (y ** 2))

    f_np_val, dx, dy = [
        f5(x, y),
        grad(lambda x, y: f5(x, y), 0)(5.0, 7.0),
        grad(lambda x, y: f5(x, y), 1)(5.0, 7.0),
    ]

    assert np.around(f_ad_val, 4) == np.around(f_np_val, 4)
    assert np.around(f_ad_grad["x"], 4) == np.around(dx, 4)
    assert np.around(f_ad_grad["y"], 4) == np.around(dy, 4)


def test_forward_class():
    variables = {"x": 3, "y": 5}
    functions = ["2*x + exp(y)", "3*x + 2*sin(y)"]
    numpy_functions = ["2*x + exp(y)", "3*x + 2*sin(y)"]
    function_ders = [
        {"x": 2, "y": np.exp(5)},
        {"x": 3, "y": 2 * np.cos(5)},
    ]  # Expected derivatives
    f = ad.Forward(variables, functions)
    for idx, variable in enumerate(f.results):
        assert variable.val == eval(
            numpy_functions[idx],
            {**variables, **{"exp": np.exp, "cos": np.cos, "sin": np.sin}},
        )
        assert compare_dicts(variable.der, function_ders[idx])


if __name__ == "__main__":
    test_add_radd()
    test_sub_rsub()
    test_mul_rmul()
    test_truediv_rtruediv()
    test_exp()
    test_ln()
    test_logistic()
    test_log10()
    test_sin()
    test_cos()
    test_tan()
    test_neg()
    test_pow()
    test_rpow()
    test_ne()
    test_lt()
    test_le()
    test_gt()
    test_ge()
    test_complicated_functions()
    test_forward_class()
