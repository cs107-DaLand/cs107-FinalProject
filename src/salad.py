import numpy as np
from utils import add_dict
import copy


class Forward(object):
    """
        For handling multiple function input
        Example:
        >> variables = {'x': 3, 'y': 5}
        >> functions = ['2*x + y', '3*x + 2*y]
        >> f = Forward(variables, functions)
        >> print(f) 
        Label: v3, Value: 11, Derivative: {'x': 2.0, 'y': 1.0}
        Label: v6, Value: 19, Derivative: {'x': 3.0, 'y': 2.0}
    """
    def __init__(self, variables: dict, functions: list):
        var_dict = {}
        for key in variables:
            var_dict[key] = Variable(val=variables[key], label=key)
        
        self.results = []
        for func in functions:
            self.results.append(eval(func, var_dict))

    def __str__(self):
        pretty = []
        for res in self.results:
            pretty.append(f"Label: {res.label}, Value: {res.val}, Derivative: {res.der}")
            pretty.append('\n')
        return ''.join(pretty).rstrip()



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
                new_dict[v] = a1 * np.asarray(dict2[v]) + a2 * np.asarray(dict1[v])
            for v in set(dict1) - set(
                dict2
            ):  # Only those variables in self DOES THIS WORK
                new_dict[v] = a2 * np.asarray(dict1[v])
            for v in set(dict2) - set(dict1):  # Only those variables in other
                new_dict[v] = a1 * np.asarray(dict2[v])
            return Variable(a1 * a2, new_dict)

        except AttributeError:
            new_dict = {}
            for key in self.der:
                new_dict[key] = other * np.asarray(self.der[key])
            return Variable(self.val * other, new_dict)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            # divide other by 1
            a2 = 1 / other.val
            dict2 = dict(other.der)
            for k in dict2:
                dict2[k] = (-1 * np.asarray(dict2[k])) / (other.val * other.val)
            reciprocal_other = Variable(a2, dict2, increment_counter=False)
            return self.__mul__(reciprocal_other)

        except AttributeError:
            # divide self by other
            a1 = self.val / other
            dict1 = dict(self.der)
            for k in dict1:
                dict1[k] = (other * np.asarray(dict1[k])) / (other * other)
            return Variable(a1, dict1)

    def __rtruediv__(self, other):
        new_var = self.__truediv__(other)
        # return reciprocal
        a2 = 1 / new_var.val
        dict2 = dict(new_var.der)
        for k in dict2:
            dict2[k] = (-1 * np.asarray(dict2[k])) / (new_var.val * new_var.val)
        reciprocal_new_var = Variable(a2, dict2, increment_counter=False)
        return reciprocal_new_var

    def __neg__(self):
        new_dir = {}
        try:
            for key in self.der:
                new_dir[key] = [-i for i in self.der[key]]
            return Variable(val=-self.val, der=new_dir, ad_mode=self.ad_mode, increment_counter=True)
        except:
            for key in self.der:
                #new_dir[key] = [-self.der[key]]
                new_dir[key] = -self.der[key]
            return Variable(val=[-self.val], der=new_dir, ad_mode=self.ad_mode, increment_counter=True)


    def __pow__(self, other):
        if isinstance(other, Variable):
            if isinstance(other.val, list) and isinstance(self.val, list):
                if not len(self.val) == len(other.val):
                    raise ValueError("Dimensions of the two variables are not equal")
                new_val = self.val ** other.val
                new_der = {}
                all_ele = set(self.der.keys()) | set(other.der.keys())
                for v in list(all_ele): # v: element
                    new_der[v] = []
                    for i in range(len(new_val)):
                        my_der = self.der[v][i] if v in self.der else 0
                        other_der = other.der[v][i] if v in other.der else 0
                        v_der = new_val[i] * (other_der * np.log(self.val[i]) + my_der / self.val[i] * other.val[i])
                        new_der[v].append(v_der)
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
            elif isinstance(other.val, list) or isinstance(self.val, list):
                raise ValueError("Dimensions of the two variables are not equal")
            else: # only number for x and y
                new_val = np.array([self.val ** other.val])
                new_der = {}
                all_ele = set(self.der.keys()) | set(other.der.keys())
                for v in list(all_ele): # v: element
                    my_der = self.der[v] if v in self.der else 0
                    other_der = other.der[v] if v in other.der else 0
                    v_der = new_val * (other_der * np.log(self.val) + my_der / self.val * other.val)
                    #new_der[v] = np.array([v_der])
                    new_der[v] = v_der
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
        else:
            try:
                new_val = self.val ** other
                new_der = {}
                for v in self.der:
                    new_der[v] = [new_val[i] * self.der[v][i] * other / self.val[i] for i in range(len(new_val))]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
            except:
                #new_val = [self.val ** other]
                new_val = self.val ** other
                new_der = {}
                for v in self.der:
                    #new_der[v] = [new_val[0] * self.der[v] * other / self.val]
                    new_der[v] = new_val * self.der[v] * other / self.val
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)


    def __rpow__(self, other):
        if isinstance(other, Variable):
            return other ** self
        else:
            try:
                new_val = other ** self.val
                new_der = {}
                for v in self.der:
                    new_der[v] = [new_val[i] * self.der[v][i] * np.log(other) for i in range(len(new_val))]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
            except:
                #new_val = [other ** self.val]
                new_val = other ** self.val
                new_der = {}
                for v in self.der:
                    #new_der[v] = [new_val[0] * self.der[v] * np.log(other)]
                    new_der[v] = new_val * self.der[v] * np.log(other)
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)

    def __ne__(self, other):
        return self.label != other.label
    
    def __lt__(self, other):
        try:
            return self.val < other.val
        except:
            if not len(self.val) == len(other.val):
                raise ValueError("Dimensions of the two variables are not equal")
            return self.val < other.val
    
    def __le__(self, other):
        try:
            return self.val <= other.val
        except:
            if not len(self.val) == len(other.val):
                raise ValueError("Dimensions of the two variables are not equal")
            return self.val <= other.val
    
    def __gt__(self, other):
        try:
            return self.val > other.val
        except:
            if not len(self.val) == len(other.val):
                raise ValueError("Dimensions of the two variables are not equal")
            return self.val > other.val
    
    def __ge__(self, other):
        try:
            return self.val >= other.val
        except:
            if not len(self.val) == len(other.val):
                raise ValueError("Dimensions of the two variables are not equal")
            return self.val >= other.val




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

def sin(x): #x is an instance of class Variable
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric sin(x)
    """
    def sin_by_element(x):
        if isinstance(x, Variable):
            val = np.sin(x.val)
            der = copy.deepcopy(x.der)
            for key in der:
                der[key] = np.cos(x.val) * x.der[key]
            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return np.sin(x)
        else:
            raise TypeError("x must be a Variable or a number")

    if isinstance(x, (list, tuple, np.ndarray)):
        return [sin_by_element(i) for i in x]
    else:
        return sin_by_element(x)

def cos(x): #x is an instance of class Variable
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric cos(x)
    """
    def cos_by_element(x):
        if isinstance(x, Variable):
            val = np.cos(x.val)
            der = copy.deepcopy(x.der)
            for key in der:
                der[key] = -np.sin(x.val) * x.der[key]
            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return np.cos(x)
        else:
            raise TypeError("x must be a Variable or a number")
    
    if isinstance(x, (list, tuple, np.ndarray)):
        return [cos_by_element(i) for i in x]
    else:
        return cos_by_element(x)

def tan(x): #x is an instance of class Variable
    """
    If x is a Variable, returns a new variable with val and der
    If x is a number, returns numeric tan(x)
    """
    def tan_by_element(x):
        if isinstance(x, Variable):
            val = np.tan(x.val)
            der = copy.deepcopy(x.der)
            for key in der:
                der[key] = 1/(np.cos(x.val)**2) * x.der[key]
            return Variable(val, der, increment_counter=True)

        elif isinstance(x, (int, float)):
            return np.tan(x)
        else:
            raise TypeError("x must be a Variable or a number")

    if isinstance(x, (list, tuple, np.ndarray)):
        return [tan_by_element(i) for i in x]
    else:
        return tan_by_element(x)
