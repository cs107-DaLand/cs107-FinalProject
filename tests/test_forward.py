import pytest
import math
import numpy as np
import autograd.numpy as adnp
from autograd import grad
import cs107_salad.Forward.salad as ad
from cs107_salad.Forward.utils import check_list, compare_dicts, compare_dicts_multi


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
    sol_val, sol_der = (
        [np.exp(3), np.exp(4), np.exp(5)],
        [np.exp(3), np.exp(4), np.exp(5),],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.exp(x) + ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [2 * np.exp(3), 2 * np.exp(4), 2 * np.exp(5)],
        [2 * np.exp(3), 2 * np.exp(4), 2 * np.exp(5),],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    z = x + x
    y = ad.exp(z)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [np.exp(2 * 3), np.exp(2 * 4), np.exp(2 * 5)],
        [2 * np.exp(2 * 3), 2 * np.exp(2 * 4), 2 * np.exp(2 * 5),],
    )
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
    sol_val, sol_der = (
        adnp.log(7),
        [
            grad(lambda x, y: adnp.log(x + y), 0)(3.0, 4.0),
            grad(lambda x, y: adnp.log(x + y), 1)(3.0, 4.0),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.ln(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [np.log(3), np.log(4), np.log(5)],
        [
            grad(lambda x: adnp.log(x))(3.0),
            grad(lambda x: adnp.log(x))(4.0),
            grad(lambda x: adnp.log(x))(5.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.ln(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.log(3 * 2), adnp.log(4 * 2), adnp.log(5 * 2)],
        [
            grad(lambda x: adnp.log(x + x))(3.0),
            grad(lambda x: adnp.log(x + x))(4.0),
            grad(lambda x: adnp.log(x + x))(5.0),
        ],
    )
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
    sol_val, sol_der = (
        logistic(7),
        [
            grad(lambda x, y: logistic(x + y), 0)(3.0, 4.0),
            grad(lambda x, y: logistic(x + y), 1)(3.0, 4.0),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.logistic(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [logistic(3), logistic(4), logistic(5)],
        [
            grad(lambda x: logistic(x))(3.0),
            grad(lambda x: logistic(x))(4.0),
            grad(lambda x: logistic(x))(5.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.logistic(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [logistic(3 * 2), logistic(4 * 2), logistic(5 * 2)],
        [
            grad(lambda x: logistic(x + x))(3.0),
            grad(lambda x: logistic(x + x))(4.0),
            grad(lambda x: logistic(x + x))(5.0),
        ],
    )
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
    sol_val, sol_der = (
        log10(7),
        [
            grad(lambda x, y: log10(x + y), 0)(3.0, 4.0),
            grad(lambda x, y: log10(x + y), 1)(3.0, 4.0),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.log10(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [log10(3), log10(4), log10(5)],
        [
            grad(lambda x: log10(x))(3.0),
            grad(lambda x: log10(x))(4.0),
            grad(lambda x: log10(x))(5.0),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([3, 4, 5], label="x")
    y = ad.log10(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [log10(3 * 2), log10(4 * 2), log10(5 * 2)],
        [
            grad(lambda x: log10(x + x))(3.0),
            grad(lambda x: log10(x + x))(4.0),
            grad(lambda x: log10(x + x))(5.0),
        ],
    )
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
    x = 0.3
    ans = ad.sin(x)
    sol = adnp.sin(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.sin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.sin(0.3), grad(adnp.sin)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.sin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.sin(0.6), grad(lambda x: adnp.sin(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.sin(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.sin(0.7),
        [
            grad(lambda x, y: adnp.sin(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.sin(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.sin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.sin(0.3), adnp.sin(0.4), adnp.sin(0.5)],
        [
            grad(lambda x: adnp.sin(x))(0.3),
            grad(lambda x: adnp.sin(x))(0.4),
            grad(lambda x: adnp.sin(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.sin(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.sin(0.3 * 2), adnp.sin(0.4 * 2), adnp.sin(0.5 * 2)],
        [
            grad(lambda x: adnp.sin(x + x))(0.3),
            grad(lambda x: adnp.sin(x + x))(0.4),
            grad(lambda x: adnp.sin(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.sin(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.sin(0.09), adnp.sin(0.10), adnp.sin(0.11)],
        [
            grad(lambda x, y: adnp.sin(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.sin(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.sin(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.sin(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.sin(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.sin(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_arcsin():
    x = 0.3
    ans = ad.arcsin(x)
    sol = adnp.arcsin(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.arcsin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.arcsin(0.3), grad(adnp.arcsin)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.arcsin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.arcsin(0.6), grad(lambda x: adnp.arcsin(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.arcsin(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.arcsin(0.7),
        [
            grad(lambda x, y: adnp.arcsin(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.arcsin(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.arcsin(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.arcsin(0.3), adnp.arcsin(0.4), adnp.arcsin(0.5)],
        [
            grad(lambda x: adnp.arcsin(x))(0.3),
            grad(lambda x: adnp.arcsin(x))(0.4),
            grad(lambda x: adnp.arcsin(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.arcsin(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.arcsin(0.3 * 2), adnp.arcsin(0.4 * 2), adnp.arcsin(0.5 * 2)],
        [
            grad(lambda x: adnp.arcsin(x + x))(0.3),
            grad(lambda x: adnp.arcsin(x + x))(0.4),
            grad(lambda x: adnp.arcsin(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.arcsin(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.arcsin(0.09), adnp.arcsin(0.10), adnp.arcsin(0.11)],
        [
            grad(lambda x, y: adnp.arcsin(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.arcsin(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.arcsin(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.arcsin(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.arcsin(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.arcsin(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)

    x = ad.Variable(2, label="x")
    with pytest.raises(Exception):
        y = ad.arcsin(x)


def test_sinh():
    x = 0.3
    ans = ad.sinh(x)
    sol = adnp.sinh(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.sinh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.sinh(0.3), grad(adnp.sinh)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.sinh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.sinh(0.6), grad(lambda x: adnp.sinh(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.sinh(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.sinh(0.7),
        [
            grad(lambda x, y: adnp.sinh(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.sinh(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.sinh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.sinh(0.3), adnp.sinh(0.4), adnp.sinh(0.5)],
        [
            grad(lambda x: adnp.sinh(x))(0.3),
            grad(lambda x: adnp.sinh(x))(0.4),
            grad(lambda x: adnp.sinh(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.sinh(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.sinh(0.3 * 2), adnp.sinh(0.4 * 2), adnp.sinh(0.5 * 2)],
        [
            grad(lambda x: adnp.sinh(x + x))(0.3),
            grad(lambda x: adnp.sinh(x + x))(0.4),
            grad(lambda x: adnp.sinh(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.sinh(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.sinh(0.09), adnp.sinh(0.10), adnp.sinh(0.11)],
        [
            grad(lambda x, y: adnp.sinh(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.sinh(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.sinh(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.sinh(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.sinh(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.sinh(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_cos():
    x = 0.3
    ans = ad.cos(x)
    sol = adnp.cos(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.cos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.cos(0.3), grad(adnp.cos)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.cos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.cos(0.6), grad(lambda x: adnp.cos(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.cos(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.cos(0.7),
        [
            grad(lambda x, y: adnp.cos(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.cos(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.cos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.cos(0.3), adnp.cos(0.4), adnp.cos(0.5)],
        [
            grad(lambda x: adnp.cos(x))(0.3),
            grad(lambda x: adnp.cos(x))(0.4),
            grad(lambda x: adnp.cos(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.cos(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.cos(0.3 * 2), adnp.cos(0.4 * 2), adnp.cos(0.5 * 2)],
        [
            grad(lambda x: adnp.cos(x + x))(0.3),
            grad(lambda x: adnp.cos(x + x))(0.4),
            grad(lambda x: adnp.cos(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.cos(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.cos(0.09), adnp.cos(0.10), adnp.cos(0.11)],
        [
            grad(lambda x, y: adnp.cos(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.cos(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.cos(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.cos(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.cos(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.cos(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_arccos():
    x = 0.3
    ans = ad.arccos(x)
    sol = adnp.arccos(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.arccos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.arccos(0.3), grad(adnp.arccos)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.arccos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.arccos(0.6), grad(lambda x: adnp.arccos(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.arccos(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.arccos(0.7),
        [
            grad(lambda x, y: adnp.arccos(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.arccos(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.arccos(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.arccos(0.3), adnp.arccos(0.4), adnp.arccos(0.5)],
        [
            grad(lambda x: adnp.arccos(x))(0.3),
            grad(lambda x: adnp.arccos(x))(0.4),
            grad(lambda x: adnp.arccos(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.arccos(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.arccos(0.3 * 2), adnp.arccos(0.4 * 2), adnp.arccos(0.5 * 2)],
        [
            grad(lambda x: adnp.arccos(x + x))(0.3),
            grad(lambda x: adnp.arccos(x + x))(0.4),
            grad(lambda x: adnp.arccos(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.arccos(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.arccos(0.09), adnp.arccos(0.10), adnp.arccos(0.11)],
        [
            grad(lambda x, y: adnp.arccos(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.arccos(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.arccos(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.arccos(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.arccos(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.arccos(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)

    x = ad.Variable(2, label="x")
    with pytest.raises(Exception):
        y = ad.arccos(x)


def test_cosh():
    x = 0.3
    ans = ad.cosh(x)
    sol = adnp.cosh(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.cosh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.cosh(0.3), grad(adnp.cosh)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.cosh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.cosh(0.6), grad(lambda x: adnp.cosh(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.cosh(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.cosh(0.7),
        [
            grad(lambda x, y: adnp.cosh(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.cosh(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.cosh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.cosh(0.3), adnp.cosh(0.4), adnp.cosh(0.5)],
        [
            grad(lambda x: adnp.cosh(x))(0.3),
            grad(lambda x: adnp.cosh(x))(0.4),
            grad(lambda x: adnp.cosh(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.cosh(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.cosh(0.3 * 2), adnp.cosh(0.4 * 2), adnp.cosh(0.5 * 2)],
        [
            grad(lambda x: adnp.cosh(x + x))(0.3),
            grad(lambda x: adnp.cosh(x + x))(0.4),
            grad(lambda x: adnp.cosh(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.cosh(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.cosh(0.09), adnp.cosh(0.10), adnp.cosh(0.11)],
        [
            grad(lambda x, y: adnp.cosh(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.cosh(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.cosh(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.cosh(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.cosh(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.cosh(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_tan():
    x = 0.3
    ans = ad.tan(x)
    sol = adnp.tan(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.tan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.tan(0.3), grad(adnp.tan)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.tan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.tan(0.6), grad(lambda x: adnp.tan(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.tan(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.tan(0.7),
        [
            grad(lambda x, y: adnp.tan(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.tan(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.tan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.tan(0.3), adnp.tan(0.4), adnp.tan(0.5)],
        [
            grad(lambda x: adnp.tan(x))(0.3),
            grad(lambda x: adnp.tan(x))(0.4),
            grad(lambda x: adnp.tan(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.tan(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.tan(0.3 * 2), adnp.tan(0.4 * 2), adnp.tan(0.5 * 2)],
        [
            grad(lambda x: adnp.tan(x + x))(0.3),
            grad(lambda x: adnp.tan(x + x))(0.4),
            grad(lambda x: adnp.tan(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.tan(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.tan(0.09), adnp.tan(0.10), adnp.tan(0.11)],
        [
            grad(lambda x, y: adnp.tan(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.tan(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.tan(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.tan(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.tan(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.tan(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_arctan():
    x = 0.3
    ans = ad.arctan(x)
    sol = adnp.arctan(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.arctan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.arctan(0.3), grad(adnp.arctan)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.arctan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.arctan(0.6), grad(lambda x: adnp.arctan(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.arctan(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.arctan(0.7),
        [
            grad(lambda x, y: adnp.arctan(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.arctan(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.arctan(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.arctan(0.3), adnp.arctan(0.4), adnp.arctan(0.5)],
        [
            grad(lambda x: adnp.arctan(x))(0.3),
            grad(lambda x: adnp.arctan(x))(0.4),
            grad(lambda x: adnp.arctan(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.arctan(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.arctan(0.3 * 2), adnp.arctan(0.4 * 2), adnp.arctan(0.5 * 2)],
        [
            grad(lambda x: adnp.arctan(x + x))(0.3),
            grad(lambda x: adnp.arctan(x + x))(0.4),
            grad(lambda x: adnp.arctan(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.arctan(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.arctan(0.09), adnp.arctan(0.10), adnp.arctan(0.11)],
        [
            grad(lambda x, y: adnp.arctan(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.arctan(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.arctan(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.arctan(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.arctan(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.arctan(x + y), 1)(0.050, 0.060),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(sol_der_x, ans_der_x) & check_list(sol_der_y, ans_der_y)


def test_tanh():
    x = 0.3
    ans = ad.tanh(x)
    sol = adnp.tanh(x)
    assert sol == ans

    x = ad.Variable(0.3, label="x")
    y = ad.tanh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.tanh(0.3), grad(adnp.tanh)(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + 0.3
    y = ad.tanh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.tanh(0.6), grad(lambda x: adnp.tanh(x + 0.3))(0.3)
    assert ans_val == sol_val
    assert math.isclose(ans_der, sol_der)

    x = ad.Variable(0.3, label="x") + ad.Variable(0.4, label="y")
    y = ad.tanh(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = (
        adnp.tanh(0.7),
        [
            grad(lambda x, y: adnp.tanh(x + y), 0)(0.3, 0.4),
            grad(lambda x, y: adnp.tanh(x + y), 1)(0.3, 0.4),
        ],
    )
    assert ans_val == sol_val
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.tanh(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.tanh(0.3), adnp.tanh(0.4), adnp.tanh(0.5)],
        [
            grad(lambda x: adnp.tanh(x))(0.3),
            grad(lambda x: adnp.tanh(x))(0.4),
            grad(lambda x: adnp.tanh(x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.3, 0.4, 0.5], label="x")
    y = ad.tanh(x + x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = (
        [adnp.tanh(0.3 * 2), adnp.tanh(0.4 * 2), adnp.tanh(0.5 * 2)],
        [
            grad(lambda x: adnp.tanh(x + x))(0.3),
            grad(lambda x: adnp.tanh(x + x))(0.4),
            grad(lambda x: adnp.tanh(x + x))(0.5),
        ],
    )
    assert check_list(ans_val, sol_val)
    assert check_list(ans_der, sol_der)

    x = ad.Variable([0.03, 0.04, 0.05], label="x")
    y = ad.Variable([0.06, 0.06, 0.06], label="y")
    y = ad.tanh(x + y)
    ans_val, ans_der_x, ans_der_y = y.val, y.der["x"], y.der["y"]
    sol_val, sol_der_x, sol_der_y = (
        [adnp.tanh(0.09), adnp.tanh(0.10), adnp.tanh(0.11)],
        [
            grad(lambda x, y: adnp.tanh(x + y), 0)(0.030, 0.060),
            grad(lambda x, y: adnp.tanh(x + y), 0)(0.040, 0.060),
            grad(lambda x, y: adnp.tanh(x + y), 0)(0.050, 0.060),
        ],
        [
            grad(lambda x, y: adnp.tanh(x + y), 1)(0.030, 0.060),
            grad(lambda x, y: adnp.tanh(x + y), 1)(0.040, 0.060),
            grad(lambda x, y: adnp.tanh(x + y), 1)(0.050, 0.060),
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

    x = ad.Variable(0, label="x")
    z = x ** 2
    assert z.val == 0
    assert z.der == {"x": 0}

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

    x = ad.Variable([0, 0], label="x")
    y = ad.Variable([1, 2], label="y")
    z = x ** y
    assert np.all(z.val == [0, 0])
    assert compare_dicts_multi(z.der, {"x": [1, 0], "y": [0, 0]}) == True


def test_rpow():
    x = ad.Variable(1, label="x")
    z = np.e ** x
    assert z.val == np.e
    assert z.der == {"x": np.e}

    x = ad.Variable(1, label="x")
    z = 0 ** x
    assert z.val == 0
    assert z.der == {"x": 0}

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

    x = ad.Variable([1, 2, 3], label="x")
    y = ad.Variable([2, 3], label="y")
    assert (x != y) == True

    z = 1
    assert (x != z) == True


def test_lt():
    x = ad.Variable(1, label="x")
    y = ad.Variable(2, label="y")
    assert (x < y) == True

    x = ad.Variable([1, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x < y) == [True, False])

    x = ad.Variable([1, 1, 1], label="x")
    y = ad.Variable([2, 2], label="y")
    with pytest.raises(Exception):
        print(x < y)


def test_le():
    x = ad.Variable(1, label="x")
    y = ad.Variable(2, label="y")
    assert (x <= y) == True

    x = ad.Variable([1, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x <= y) == [True, True])

    x = ad.Variable([1, 1, 1], label="x")
    y = ad.Variable([2, 2], label="y")
    with pytest.raises(Exception):
        print(x <= y)


def test_gt():
    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    assert (x > y) == True

    x = ad.Variable([3, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x > y) == [True, False])

    x = ad.Variable([1, 1, 1], label="x")
    y = ad.Variable([2, 2], label="y")
    with pytest.raises(Exception):
        print(x > y)


def test_ge():
    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    assert (x >= y) == True

    x = ad.Variable([3, 2], label="x")
    y = ad.Variable([2, 2], label="y")
    assert np.all((x >= y) == [True, True])

    x = ad.Variable([1, 1, 1], label="x")
    y = ad.Variable([2, 2], label="y")
    with pytest.raises(Exception):
        print(x >= y)


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

    ## Function 6
    ## arctan(tan(2*x-y/2) * y - exp(x^2 / y^2))
    x = 1
    x_var = ad.Variable(x, label="x")
    y = -1
    y_var = ad.Variable(y, label="y")

    f_ad = ad.arctan(
        ad.tan(2 * x_var - y_var / 2) * y_var - ad.exp(x_var ** 2 / y_var ** 2)
    )
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    def f6(x, y):
        return adnp.arctan(adnp.tan(2 * x - y / 2) * y - adnp.exp((x ** 2) / (y ** 2)))

    f_np_val, dx, dy = [
        f6(x, y),
        grad(lambda x, y: f6(x, y), 0)(1.0, -1.0),
        grad(lambda x, y: f6(x, y), 1)(1.0, -1.0),
    ]

    assert np.around(f_ad_val, 4) == np.around(f_np_val, 4)
    assert np.around(f_ad_grad["x"], 4) == np.around(dx, 4)
    assert np.around(f_ad_grad["y"], 4) == np.around(dy, 4)

    ## Function 7
    ## arcsin(cos(tan(2*x-y/2) * y - exp(x^2 / y^2)))
    x = 1
    x_var = ad.Variable(x, label="x")
    y = -1
    y_var = ad.Variable(y, label="y")

    f_ad = ad.arcsin(
        ad.cos(ad.tan(2 * x_var - y_var / 2) * y_var - ad.exp(x_var ** 2 / y_var ** 2))
    )
    f_ad_val = f_ad.val
    f_ad_grad = f_ad.der

    def f7(x, y):
        return adnp.arcsin(
            adnp.cos(adnp.tan(2 * x - y / 2) * y - adnp.exp((x ** 2) / (y ** 2)))
        )

    f_np_val, dx, dy = [
        f7(x, y),
        grad(lambda x, y: f7(x, y), 0)(1.0, -1.0),
        grad(lambda x, y: f7(x, y), 1)(1.0, -1.0),
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


def test_str():
    x = np.random.rand(10, 10)
    x_var = ad.Variable(x, label="x")
    assert (
        str(x_var)
        == f"Label: {x_var.label}, Value: {x_var.val}, Derivative: {x_var.der}"
    )

    variables = {"x": 1}
    functions = ["x"]
    f = ad.Forward(variables, functions)
    assert (
        str(f)
        == f"Function: {f.functions[0]}, Value: {f.results[0].val}, Derivative: {f.results[0].der}"
    )


def test_sqrt():
    x = ad.Variable(3, label="x")
    z = ad.sqrt(x)
    assert z.val == np.sqrt(3)
    assert z.der == {"x": 1 / (2 * np.sqrt(x.val))}

    x = ad.Variable([3, 2], label="x")
    z = ad.sqrt(x)
    assert np.all(z.val == np.sqrt(x.val))
    assert np.all(z.der["x"] == 1 / (2 * np.sqrt(x.val)))

    x = ad.Variable(3, label="x")
    y = ad.Variable(2, label="y")
    z = ad.sqrt(x * y)
    assert z.val == np.sqrt(6)
    assert z.der == {
        "x": y.val / (2 * np.sqrt(x.val * y.val)),
        "y": x.val / (2 * np.sqrt(x.val * y.val)),
    }

    x = ad.Variable([3, 2], label="x")
    y = ad.Variable([2, 3], label="y")
    z = ad.sqrt(x * y)
    assert np.all(z.val == np.sqrt(x.val * y.val))
    assert compare_dicts_multi(
        z.der,
        {
            "x": y.val / (2 * np.sqrt(x.val * y.val)),
            "y": x.val / (2 * np.sqrt(x.val * y.val)),
        },
    )

    x = ad.Variable(-1, label="x")
    with pytest.raises(Exception):
        z = ad.sqrt(x)


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
    test_str()
    test_arcsin()
    test_arccos()
    test_arctan()
    test_sinh()
    test_cosh()
    test_tanh()
    test_sqrt()
