import pytest
import numpy as np
import cs107_salad.Forward.salad as ad
import cs107_salad.Optimization.optimize as optimize


def test_gradient_descent():

    f = "sin(x**2)"
    starting_pos = {"x": 5}

    GD = optimize.GradientDescent()
    min_params, val, der = GD.optimize(f, starting_pos)
    assert der["x"] == pytest.approx(0, abs=1e-5)
    assert np.sin(min_params["x"] ** 2) == pytest.approx(-1, abs=1e-5)
    assert val == pytest.approx(-1, abs=1e-5)

    f = "sin(x**2+y**2)"
    starting_pos = {"x": 5, "y": 5}

    GD = optimize.GradientDescent()
    min_params, val, der = GD.optimize(f, starting_pos)
    assert np.sin(min_params["x"] ** 2 + min_params["y"] ** 2) == pytest.approx(
        -1, abs=1e-5
    )
    assert val == pytest.approx(-1, abs=1e-5)

    f = "2 * x**2"
    starting_pos = {"x": 5.0}

    GD = optimize.GradientDescent()
    min_params, val, der, hist = GD.optimize(f, starting_pos, full_history=True)
    hist["x"][-1] == pytest.approx(min_params["x"], abs=1e-5)
    assert min_params["x"] == pytest.approx(0, abs=1e-5)
    assert val == pytest.approx(0, abs=1e-5)

    f = "x**2 + y**2"
    starting_pos = {"x": 5, "y": 10}

    GD = optimize.GradientDescent()
    min_params, val, der = GD.optimize(f, starting_pos)
    assert min_params["x"] == pytest.approx(0, abs=1e-5)
    assert min_params["y"] == pytest.approx(0, abs=1e-5)
    assert val == pytest.approx(0, abs=1e-5)


def test_BFGS():

    f = "sin(x**2)"
    starting_pos = {"x": 5}

    BFGS = optimize.BFGS()
    min_params, val = BFGS.optimize(f, starting_pos)
    assert np.sin(min_params["x"] ** 2) == pytest.approx(-1, abs=1e-5)
    assert val == pytest.approx(-1, abs=1e-5)

    f = "sin(3*3.1415926/2 + x**2+y**2)"
    starting_pos = {"x": 1, "y": 1}

    BFGS = optimize.BFGS()
    min_params, val, hist = BFGS.optimize(
        f, starting_pos, full_history=True, max_iter=50
    )
    for i in range(len(hist["x"])):
        print("x:", hist["x"][i])
        print("y:", hist["y"][i])
        print("f:", np.sin(3 * 3.1415926 / 2 + hist["x"][i] ** 2 + hist["y"][i] ** 2))
    assert np.sin(
        3 * 3.1415926 / 2 + min_params["x"] ** 2 + min_params["y"] ** 2
    ) == pytest.approx(-1, abs=1e-5)
    assert val == pytest.approx(-1, abs=1e-5)

    f = "2 * x**2"
    starting_pos = {"x": 5.0}

    BFGS = optimize.BFGS()
    min_params, val, hist = BFGS.optimize(f, starting_pos, full_history=True)
    hist["x"][-1] == pytest.approx(min_params["x"], abs=1e-5)
    assert min_params["x"] == pytest.approx(0, abs=1e-5)
    assert val == pytest.approx(0, abs=1e-5)

    f = "x**2 + y**2"
    starting_pos = {"x": 5, "y": 10}

    BFGS = optimize.BFGS()
    min_params, val = BFGS.optimize(f, starting_pos)
    assert min_params["x"] == pytest.approx(0, abs=1e-5)
    assert min_params["y"] == pytest.approx(0, abs=1e-5)
    assert val == pytest.approx(0, abs=1e-5)

    f = "100 * (y-x**2)**2 + (1-x)**2"
    starting_pos = {"x": -1, "y": 1}

    BFGS = optimize.BFGS()
    min_params, val = BFGS.optimize(f, starting_pos)
    assert min_params["x"] == pytest.approx(1, abs=1e-5)
    assert min_params["y"] == pytest.approx(1, abs=1e-5)
    assert val == pytest.approx(0, abs=1e-5)


def test_SGD():
    X = np.random.rand(100, 3)
    y = X @ np.array([0, 1, 2])

    SGD = optimize.StochasticGradientDescent(X, y, batch_size=10)
    min_params, val, der = SGD.optimize([0, 0, 0], max_iter=5000, learning_rate=0.01)
    assert min_params["b0"] == pytest.approx(0, abs=1e-2)
    assert min_params["b1"] == pytest.approx(1, abs=1e-2)
    assert min_params["b2"] == pytest.approx(2, abs=1e-2)

    X = np.random.rand(100, 3)
    y = X @ np.array([2, 0, 3])

    SGD = optimize.StochasticGradientDescent(X, y, batch_size=10)
    min_params, val, der, hist = SGD.optimize(
        [0, 0, 0], max_iter=5000, learning_rate=0.01, full_history=True
    )
    assert min_params["b0"] == pytest.approx(2, abs=1e-2)
    assert min_params["b1"] == pytest.approx(0, abs=1e-2)
    assert min_params["b2"] == pytest.approx(3, abs=1e-2)
    assert hist["b0"][-1] == min_params["b0"]
    assert hist["b1"][-1] == min_params["b1"]
    assert hist["b2"][-1] == min_params["b2"]


if __name__ == "__main__":
    test_gradient_descent()
    test_BFGS()
    test_SGD()
