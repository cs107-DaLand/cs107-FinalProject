import numpy as np
from utils import add_dict
import copy


class Variable(object):
    counter = 0

    def __init__(
        self, val, der=None, label=None, ad_mode="forward", increment_counter=True
    ):

        self.val = np.asarray(val)
        self.ad_mode = ad_mode

        if label is not None:
            self.label = label
        else:
            self.label = "v" + Variable.get_counter()

        if der is not None:
            self.der = der
        else:
            self.der = {self.label: np.ones(self.val.shape)}

        if increment_counter:
            Variable.increment()

    def __str__(self):
        return f"Label: {self.label}, Value: {self.val}, Derivative: {self.der}"

    @staticmethod
    def increment():
        Variable.counter += 1

    @staticmethod
    def get_counter():
        return str(Variable.counter)

    def __add__(self, other):
        try:
            return Variable(self.val + other.val, add_dict(self.der, other.der))
        except AttributeError:
            return Variable(self.val + other, self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        try:
            a1, a2 = self.val, other.val
            dict1, dict2 = self.der, other.der
            new_dict = {}
            for v in set(dict1).intersection(set(dict2)):  # intersection
                new_dict[v] = a1 * dict2[v] + a2 * dict1[v]
            for v in set(dict1) - set(
                dict2
            ):  # Only those variables in self DOES THIS WORK
                new_dict[v] = a2 * dict1[v]
            for v in set(dict2) - set(dict1):  # Only those variables in other
                new_dict[v] = a1 * dict2[v]
            return Variable(a1 * a2, new_dict)

        except AttributeError:
            new_dict = {}
            for key in self.der:
                new_dict[key] = other * self.der[key]
            return Variable(self.val * other, new_dict)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            # divide other by 1
            a2 = 1 / other.val
            dict2 = dict(other.der)
            for k in dict2:
                dict2[k] = (-1 * dict2[k]) / (other.val * other.val)
            reciprocal_other = Variable(a2, dict2, increment_counter=False)
            return self.__mul__(reciprocal_other)

        except AttributeError:
            # divide self by other
            a1 = self.val / other
            dict1 = dict(self.der)
            for k in dict1:
                dict1[k] = (other * dict1[k]) / (other * other)
            return Variable(a1, dict1)

    def __rtruediv__(self, other):
        new_var = self.__truediv__(other)
        # return reciprocal
        a2 = 1 / new_var.val
        dict2 = dict(new_var.der)
        for k in dict2:
            dict2[k] = (-1 * dict2[k]) / (new_var.val * new_var.val)
        reciprocal_new_var = Variable(a2, dict2, increment_counter=False)
        return reciprocal_new_var


def exp(x):
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric exp(x)
    """

    def exp_by_element(x):
        if isinstance(x, Variable):
            val = np.exp(x.val)
            der = copy.deepcopy(x.der)
            for key in der:
                der[key] = val * x.der[key]

            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return np.exp(x)
        else:
            raise TypeError("x must be a Variable or a number")

    if isinstance(x, (list, tuple, np.ndarray)):
        return [exp_by_element(i) for i in x]
    else:
        return exp_by_element(x)


def log10(x):
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric log(x)
    """

    def log10_by_element(x):
        if isinstance(x, Variable):
            val = np.log(x.val) / np.log(10)
            der = copy.deepcopy(x.der)
            for key in der:
                der[key] = 1 / x.val * 1 / np.log(10) * x.der[key]
            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return np.log(x) / np.log(10)
        else:
            raise TypeError("x must be a Variable or a number")

    if isinstance(x, (list, tuple, np.ndarray)):
        return [log10_by_element(i) for i in x]
    else:
        return log10_by_element(x)

    # 1/x * 1/log(10)


def ln(x):
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric ln(x)
    """

    def ln_by_element(x):
        if isinstance(x, Variable):
            val = np.log(x.val)
            der = copy.deepcopy(x.der)
            for key in der:
                der[key] = 1 / x.val * x.der[key]
            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return np.log(x)
        else:
            raise TypeError("x must be a Variable or a number")

    if isinstance(x, (list, tuple, np.ndarray)):
        return [ln_by_element(i) for i in x]
    else:
        return ln_by_element(x)


def logistic(x):
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric 1 / (1 + np.exp(x))
    """

    def logistic_by_element(x):
        if isinstance(x, Variable):
            val = 1 / (1 + np.exp(-x.val))
            der = copy.deepcopy(x.der)
            for key in der:
                # derivative of logistic function is logistic(x) * (1 - logistic(x))
                der[key] = val * (1 - val) * x.der[key]
            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return 1 / (1 + np.exp(-x))
        else:
            raise TypeError("x must be a Variable or a number")

    if isinstance(x, (list, tuple, np.ndarray)):
        return [logistic_by_element(i) for i in x]
    else:
        return logistic_by_element(x)
