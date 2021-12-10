from numpy.core.numeric import full
from scipy.optimize.optimize import OptimizeResult
from ..Forward import salad as ad
import numpy as np
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, learning_rate=0.1, max_iter=5000, grad_tol=1e-6):
        pass

    def optimize(self):
        pass


class GradientDescent(Optimizer):
    """
    Class to run GD on a function

    Attributes
    ----------
    f: function, str
        The function to minimize.
    min_params: a dict of the form {var: value}
        The optimized parameters
    """

    def __init__(self):
        pass

    def optimize(
        self,
        f,
        starting_pos,
        learning_rate=0.01,
        grad_tol=1e-8,
        max_iter=5000,
        full_history=False,
    ):
        """
        Minimize a function using gradient descent.

        Parameters
        ----------
        f: function, str
            The function to minimize.
        starting_pos: a dict of the form {var: value}
            The starting parameters
        learning_rate: float
            The learning rate for the gradient descent algorithm
        grad_tol: float
            The tolerance for the gradient
        max_iter: int
            The maximum number of iterations to run the gradient descent algorithm
        full_history: bool
            Whether to return the full history of the gradient descent algorithm

        Returns
        -------
        self.min_params: a dict of the form {var: value}
            The optimized parameters
        val: (list of) float
            The value of the function at the minimum
        der: a dict of the form {var: derivative}
            The derivative of the function at the optimized parameters
        history: bool
            Whether to return the full history of the gradient descent algorithm
        
        Examples
        --------
        >>> f = "sin(x**2)"
        >>> starting_pos = {"x": 5}
        >>> GD = optimize.GradientDescent()
        >>> min_params, der = GD.optimize(f, starting_pos)
        >>> print(min_params)
        {'x': 4.854064781389672}
        >>> print(der)
        {'x': 6.942271683968039e-10}
        """
        self.learning_rate = learning_rate
        self.func = f
        # check starting_pos is in the correct format
        if not isinstance(starting_pos, dict):
            raise ValueError("starting_pos must be a dict of the form {var: value}")

        self.min_params = {s: v for s, v in starting_pos.items()}
        self.history = {s: [v] for s, v in starting_pos.items()}

        for i in range(max_iter):
            ad_f = ad.Forward(self.min_params, [self.func])

            for idx, variable in enumerate(ad_f.results):
                der = {s: v for s, v in variable.der.items()}
            # update params
            for param in self.min_params.keys():
                self.min_params[param] -= learning_rate * der[param]
                if full_history:
                    self.history[param].append(self.min_params[param])

            # check if gradient is small enough
            grad_length = np.linalg.norm(np.array(list(der.values())))
            if grad_length < grad_tol:
                break
        # calculate the value of the function at the minimum
        ad_f = ad.Forward(self.min_params, [self.func])
        for idx, variable in enumerate(ad_f.results):
            val = variable.val

        if full_history:
            return self.min_params, val, der, self.history
        else:
            return self.min_params, val, der


class BFGS(Optimizer):
    """
    Class to run BFGS on a function

    Attributes
    ----------
    f: function, str
        The function to minimize.
    min_params: a dict of the form {var: value}
        The optimized parameters
    """

    def __init__(self):
        pass

    def optimize(
        self, f, starting_pos, tol=1e-8, max_iter=5000, full_history=False,
    ):
        """
        Minimize a function using gradient descent.

        Parameters
        ----------
        f: function, str
            The function to minimize.
        starting_pos: a dict of the form {var: value}
            The starting parameters
        tol: float
            The tolerance for the gradient
        max_iter: int
            The maximum number of iterations to run the gradient descent algorithm
        full_history: bool
            Whether to return the full history of the gradient descent algorithm
        
        Returns
        -------
        self.min_params: a dict of the form {var: value}
            The optimized parameters
        val: (list of) float
            The value of the function at the minimum
        history: bool
            Whether to return the full history of the gradient descent algorithm
        
        Examples
        --------
        >>> f = "sin(x**2)"
        >>> starting_pos = {"x": 5}
        >>> BFGS = optimize.BFGS()
        >>> min_params = BFGS.optimize(f, starting_pos)
        >>> print(min_params)
        {'x': -2.1708037636748028}
        """
        self.func = f
        self.params = (
            starting_pos.keys()
        )  # this is to keep track of the order of the parameters
        self.B = np.eye(len(starting_pos))
        # check starting_pos is in the correct format
        if not isinstance(starting_pos, dict):
            raise ValueError("starting_pos must be a dict of the form {var: value}")

        # initialize the parameters
        self.min_params = {param: starting_pos[param] for param in self.params}
        self.history = {param: [starting_pos[param]] for param in self.params}

        for i in range(max_iter):
            # compute gradient
            ad_f_k = ad.Forward(self.min_params, [self.func])
            for idx, variable in enumerate(ad_f_k.results):
                der_k = {param: variable.der[param] for param in self.params}
            grad_fk = np.array([der_k[param] for param in self.params])  # grad f(x_k)
            x_k = np.array([self.min_params[param] for param in self.params])  # x_k
            s_k = -np.linalg.solve(self.B, grad_fk)  # s_k

            x_kp1 = x_k + s_k  # x_{k+1}
            self.min_params = {
                param: x_kp1[idx] for idx, param in enumerate(self.params)
            }  # update params

            if full_history:  # update history
                for param in self.params:
                    self.history[param].append(self.min_params[param])  # update history

            if np.linalg.norm(s_k) < tol:  # if update is small enough, break
                break

            # compute gradient of f(x_{k+1})
            ad_f_kp1 = ad.Forward(self.min_params, [self.func])
            for idx, variable in enumerate(ad_f_kp1.results):
                der_kp1 = {param: variable.der[param] for param in self.params}
            grad_fkp1 = np.array(
                [der_kp1[param] for param in self.params]
            )  # grad f(x_{k+1})
            y_k = grad_fkp1 - grad_fk  # y_k

            # reshape vectors
            s_k = s_k.reshape(-1, 1)
            x_k = x_k.reshape(-1, 1)
            x_kp1 = x_kp1.reshape(-1, 1)
            y_k = y_k.reshape(-1, 1)
            grad_fk = grad_fk.reshape(-1, 1)
            grad_fkp1 = grad_fkp1.reshape(-1, 1)

            der_B = (y_k @ y_k.T) / (y_k.T @ s_k) - (self.B @ s_k @ s_k.T @ self.B) / (
                s_k.T @ self.B @ s_k
            )  # y_k
            # update B
            self.B = self.B + der_B  # B_{k+1}

        # calculate the value of the function at the minimum
        ad_f = ad.Forward(self.min_params, [self.func])
        for idx, variable in enumerate(ad_f.results):
            val = variable.val

        if full_history:
            return self.min_params, val, self.history
        else:
            return self.min_params, val


class StochasticGradientDescent(Optimizer):
    """
    Class to run Mean Squared Error Minimization on a dataset using gradient descent
    """

    def __init__(self, X, y, batch_size=10):
        """
        Minimize a function using stochastic gradient descent.

        Parameters
        ----------
        X: numpy.ndarray
            The dataset
        y: numpy.ndarray
            The labels
        batch_size: int
            The size of the batch
        """

        self.data = X
        self.target = y
        self.n_predictors = self.data.shape[1]
        self.n_obs = self.data.shape[0]
        self.batch_size = batch_size  # in terms of number of observations
        self.params = [f"b{idx}" for idx in range(self.n_predictors)]

        assert (
            self.batch_size <= self.n_obs
        )  # batch size cannot be larger than the number of observations
        assert self.batch_size > 0  # batch size must be positive
        assert type(self.batch_size) == int

    def optimize(
        self,
        starting_pos,
        learning_rate=0.01,
        grad_tol=1e-8,
        max_iter=5000,
        full_history=False,
    ):
        """
        Minimize a function using stochastic gradient descent.
        
        Parameters
        ----------
        starting_pos: a dict of the form {var: value}
            The starting parameters
        learning_rate: float
            The learning rate
        grad_tol: float
            The tolerance for the gradient
        max_iter: int
            The maximum number of iterations to run the gradient descent algorithm
        full_history: bool
            Whether to return the full history of the gradient descent algorithm

        Returns
        -------
        self.min_params: a dict of the form {var: value}
            The optimized parameters
        val: (list of) float
            The value of the function at the minimum
        der: a dict of the form {var: derivative}
            The derivative of the function at the optimized parameters

        Examples
        --------
        >>> import numpy as np
        >>> X = np.random.rand(100, 3)
        >>> y = X @ np.array([0, 1, 2])
        >>> SGD = optimize.StochasticGradientDescent(X, y, batch_size=10)
        >>> min_params, der = SGD.optimize([0, 0, 0], max_iter=5000, learning_rate=0.01)
        >>> print(min_params)
        {'b0': 0.00038398377190103383,
         'b1': 1.0002070838383432,
         'b2': 1.999274296181649}
        >>> print(der)
        {'b0': -1.88086866634477e-06,
         'b1': -0.00010693649472573479,
         'b2': -0.0001979083464866652}
        """

        self.learning_rate = learning_rate
        self.grad_tol = grad_tol
        assert (
            len(starting_pos) == self.data.shape[1]
        )  # starting_pos must have the same length as the number of parameters

        # initialize the parameters
        self.min_params = {
            param: starting_pos[idx] for idx, param in enumerate(self.params)
        }
        if full_history:
            self.history = {
                param: [starting_pos[idx]] for idx, param in enumerate(self.params)
            }

        for i in range(max_iter):

            # get minibatch
            minibatch_idx = self._get_batch()
            minibatch = self.data[minibatch_idx]
            minibatch_target = self.target[minibatch_idx]

            # formulat the objective function
            self.func = ""
            for i in range(len(minibatch)):
                self.func += f"( {minibatch_target[i]} - "
                for idx, param in enumerate(self.params):
                    self.func += f"{param}*{minibatch[i,idx]} - "
                self.func = self.func[:-3]
                self.func += f")**2 / {len(minibatch)} + "
            self.func = self.func[:-3]

            # call GD with one iteration on the minibatch
            GD = GradientDescent()
            self.min_params, _, der = GD.optimize(
                self.func,
                self.min_params,
                max_iter=1,
                grad_tol=self.grad_tol,
                learning_rate=self.learning_rate,
            )
            if full_history:
                for param in self.params:
                    self.history[param].append(self.min_params[param])

            # check if gradient is small enough
            grad_length = np.linalg.norm(np.array(list(der.values())))
            if grad_length < grad_tol:
                break

        # calculate the value of the function at the minimum
        ad_f = ad.Forward(self.min_params, [self.func])
        for idx, variable in enumerate(ad_f.results):
            val = variable.val

        if full_history:
            return self.min_params, val, der, self.history
        else:
            return self.min_params, val, der

    def _get_batch(self):
        """
        Helper function toeturns a random batch of indices
        """
        idx = np.random.choice(len(self.data), self.batch_size, replace=False)
        return idx

