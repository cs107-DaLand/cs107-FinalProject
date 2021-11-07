import pytest
import math
import numpy as np
import autograd.numpy as adnp
from autograd import grad
import salad as ad

## For comparing derivative dictionaries with rounding
def compare_dicts(dict1, dict2, round_place=4):
    for k in dict2:
        if np.round(dict1[k], round_place) != np.round(dict2[k], round_place):
            return False
    return True


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
    print("-" * 50, "Test exp")
    x = 3
    ans = ad.exp(x)
    sol = np.exp(x)
    print("ad.exp(3)")
    print(f"val match? {sol == ans}")

    x = ad.Variable(3, label="x")
    y = ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = np.exp(3), np.exp(3)
    print("ad.exp(var(3))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x")
    ans_val, ans_der = ad.exp(x).val, ad.exp(x).der["x"]
    sol_val, sol_der = np.exp(3), np.exp(3)
    print("ad.exp(var(3))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + 3
    y = ad.exp(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = np.exp(6), np.exp(6)
    print("ad.exp(var(3)+3)")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.exp(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = np.exp(7), [np.exp(7), np.exp(7)]
    print("ad.exp(var(3) + var(4))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der[0], sol_der[0], rel_tol=1e-9, abs_tol=0.0) & math.isclose(ans_der[1], sol_der[1], rel_tol=1e-9, abs_tol=0.0)}"
    )


def test_ln():
    print("-" * 50, "Test ln")
    x = 3
    ans = ad.ln(x)
    sol = adnp.log(x)
    print("ad.ln(3)")
    print(f"val match? {sol == ans}")

    x = ad.Variable(3, label="x")
    y = ad.ln(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.log(3), grad(adnp.log)(3.0)
    print("ad.ln(var(3))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + 3
    y = ad.ln(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = adnp.log(6), grad(lambda x: adnp.log(x + 3.0))(3.0)
    print("ad.ln(var(3)+3)")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.ln(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = adnp.log(7), [
        grad(lambda x, y: adnp.log(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: adnp.log(x + y), 1)(3.0, 4.0),
    ]
    print("ad.ln(var(3) + var(4))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der[0], sol_der[0], rel_tol=1e-9, abs_tol=0.0) & math.isclose(ans_der[1], sol_der[1], rel_tol=1e-9, abs_tol=0.0)}"
    )


def test_logistic():
    print("-" * 50, "Test logistic")

    def logistic(x):
        return 1 / (1 + adnp.exp(-x))

    x = 3
    ans = ad.logistic(x)
    sol = logistic(x)
    print("ad.logistic(3)")
    print(f"val match? {sol == ans}")

    x = ad.Variable(3, label="x")
    y = ad.logistic(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = logistic(3), grad(logistic)(3.0)
    print("ad.logistic(var(3))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + 3
    y = ad.logistic(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = logistic(6), grad(lambda x: logistic(x + 3.0))(3.0)
    print("ad.logistic(var(3)+3)")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.logistic(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = logistic(7), [
        grad(lambda x, y: logistic(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: logistic(x + y), 1)(3.0, 4.0),
    ]
    print("ad.logistic(var(3) + var(4))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der[0], sol_der[0], rel_tol=1e-9, abs_tol=0.0) & math.isclose(ans_der[1], sol_der[1], rel_tol=1e-9, abs_tol=0.0)}"
    )


def test_log10():
    print("-" * 50, "Test log10")

    def log10(x):
        return adnp.log(x) / adnp.log(10)

    x = 3
    ans = ad.log10(x)
    sol = log10(x)
    print("ad.log10(3)")
    print(f"val match? {sol == ans}")

    x = ad.Variable(3, label="x")
    y = ad.log10(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = log10(3), grad(log10)(3.0)
    print("ad.log10(var(3))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + 3
    y = ad.log10(x)
    ans_val, ans_der = y.val, y.der["x"]
    sol_val, sol_der = log10(6), grad(lambda x: log10(x + 3.0))(3.0)
    print("ad.log10(var(3)+3)")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der, sol_der, rel_tol=1e-9, abs_tol=0.0)}"
    )

    x = ad.Variable(3, label="x") + ad.Variable(4, label="y")
    y = ad.log10(x)
    ans_val, ans_der = y.val, [y.der["x"], y.der["y"]]
    sol_val, sol_der = log10(7), [
        grad(lambda x, y: log10(x + y), 0)(3.0, 4.0),
        grad(lambda x, y: log10(x + y), 1)(3.0, 4.0),
    ]
    print("ad.log10(var(3) + var(4))")
    print(
        f"val match? {ans_val == sol_val}; der match? {math.isclose(ans_der[0], sol_der[0], rel_tol=1e-9, abs_tol=0.0) & math.isclose(ans_der[1], sol_der[1], rel_tol=1e-9, abs_tol=0.0)}"
    )


if __name__ == "__main__":
    test_add_radd()
    test_sub_rsub()
    test_mul_rmul()
    test_truediv_rtruediv()
    test_exp()
    test_ln()
    test_logistic()
    test_log10()
