import numpy as np
from .utils import add_dict, compare_dicts_multi
import copy


class Forward(object):
    """
    For handling multiple function input
    
    Attributes
    ----------
    results : list
        List of results (Variable) of different functions
    functions : list
        List of functions
    """
    def __init__(self, variables: dict, functions: list):
        """
        Parameters
        ----------
        variables : dict
            Dictionary of variables
        functions : list
            List of functions
        
        Examples
        -------
        >>> variables = {'x': 3, 'y': 5}
        >>> functions = ['2*x + y', '3*x + 2*y']
        >>> f = Forward(variables, functions)
        >>> print(f) 
        Function: 2*x + y, Label: v3, Value: 11, Derivative: {'x': 2.0, 'y': 1.0}
        Function: 3*x + 2*y, Label: v6, Value: 19, Derivative: {'x': 3.0, 'y': 2.0}
        """
        # need to overwrite functions
        var_dict = {
            'tan': tan,
            'cos': cos,
            'sin': sin,
            'tan': tan,
            'tan': tan,
            "logistic": logistic,
            "ln": ln,
            "log10": log10,
            "exp": exp
        }
        # var_dict = {}
        for key in variables:
            var_dict[key] = Variable(val=variables[key], label=key)
        
        self.results = []
        for func in functions:
            self.results.append(eval(func, var_dict))
        
        self.functions = functions

    def __str__(self):
        """
        Prints the results of the functions

        Returns
        -------
        str
            Description of results      
        """
        pretty = []
        for idx, res in enumerate(self.results):
            pretty.append(f"Function: {self.functions[idx]}, Value: {res.val}, Derivative: {res.der}")
            pretty.append('\n')
        return ''.join(pretty).rstrip()



class Variable(object):
    """
    Create a Variable object for automatic differentiation.

    Attributes
    ----------
    val : float
        Value of the variable
    der : dict
        Dictionary of derivatives
    label : str
        Label of the variable
    ad_mode : str
        Automatic differentiation mode (only forward for now)
    counter : int
        Counter for the label
    """
    counter = 0

    def __init__(self, val, der=None, label=None, ad_mode="forward", increment_counter=True):
        """
        Parameters
        ----------
        val : float
            Value of the variable
        der : dict
            Dictionary of derivatives
        label : str
            Label of the variable
        ad_mode : str
            Automatic differentiation mode (only forward for now)
        increment_counter : bool
            Increment the counter for the label
        
        Examples
        -------
        >>> v = Variable(3, label='x')
        >>> print(v.val)
        3
        >>> print(v.der)
        {'x': 1.0}

        >>> v = Variable([1,2], {'x': [1,2]})
        >>> print(v.val)
        [1 2]
        >>> print(v.der)
        {'x': [1 2]}

        >>> v = Variable([1,2], {'x': [1,2], 'y': [3,4]})
        >>> print(v.val)
        [1 2]
        >>> print(v.der)
        {'x': [1 2], 'y': [3 4]}
        """

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
        """
        Prints the value and the derivatives of the variable

        Returns
        -------
        str
            Description of the variable
        """
        return f"Label: {self.label}, Value: {self.val}, Derivative: {self.der}"

    @staticmethod
    def increment():
        Variable.counter += 1

    @staticmethod
    def get_counter():
        return str(Variable.counter)

    def __add__(self, other):
        """
        Calculates the sum of two variables

        Parameters
        ----------
        other : Variable
            Variable to be added
        
        Returns
        -------
        Variable
            Variable of the sum
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = Variable(5, label = 'y')
        >>> v3 = v1 + v2
        >>> print(v3.val)
        8
        >>> print(v3.der)
        {'x': 1.0, 'y': 1.0}
        """
        try:
            return Variable(self.val + other.val, add_dict(self.der, other.der))
        except AttributeError:
            return Variable(self.val + other, self.der)

    def __radd__(self, other):
        """
        Calculates the sum of two variables

        Parameters
        ----------
        other : Variable
            Variable to be added
        
        Returns
        -------
        Variable
            Variable of the sum
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = 3 + v1
        >>> print(v2.val)
        6
        >>> print(v2.der)
        {'x': 1.0}
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Calculates the difference of two variables

        Parameters
        ----------
        other : Variable
            Variable to be subtracted
        
        Returns
        -------
        Variable
            Variable of the difference
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = Variable(5, label = 'y')
        >>> v3 = v1 - v2
        >>> print(v3.val)
        -2
        >>> print(v3.der)
        {'x': 1.0, 'y': -1.0}
        """
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        """
        Calculates the difference of two variables

        Parameters
        ----------
        other : Variable
            Variable to be subtracted
        
        Returns
        -------
        Variable
            Variable of the difference
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = 5 - v1
        >>> print(v2.val)
        -2
        >>> print(v2.der)
        {'x': -1.0}
        """
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        """
        Calculates the product of two variables

        Parameters
        ----------
        other : Variable
            Variable to be multiplied
        
        Returns
        -------
        Variable
            Variable of the product
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = Variable(5, label = 'y')
        >>> v3 = v1 * v2
        >>> print(v3.val)
        15
        >>> print(v3.der)
        {'x': 5.0, 'y': 3.0}
        """
        try:
            a1, a2 = self.val, other.val
            dict1, dict2 = self.der, other.der
            new_dict = {}
            for v in set(dict1).intersection(set(dict2)):  # intersection
                new_dict[v] = a1 * np.asarray(dict2[v]) + a2 * np.asarray(dict1[v])
            for v in set(dict1) - set(dict2):  # Only those variables in self
                new_dict[v] = a2 * np.asarray(dict1[v])
            for v in set(dict2) - set(dict1):  # Only those variables in other
                new_dict[v] = a1 * np.asarray(dict2[v])
            return Variable(a1 * a2, new_dict)

        except AttributeError:
            new_dict = {}
            for key in self.der:
                new_dict[key] = np.asarray(other * np.asarray(self.der[key]))
            return Variable(self.val * other, new_dict)

    def __rmul__(self, other):
        """
        Calculates the product of two variables

        Parameters
        ----------
        other : Variable
            Variable to be multiplied
        
        Returns
        -------
        Variable
            Variable of the product
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = 3 * v1
        >>> print(v2.val)
        9
        >>> print(v2.der)
        {'x': 3.0}
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Calculates the division of two variables

        Parameters
        ----------
        other : Variable
            Variable to be divided
        
        Returns
        -------
        Variable
            Variable of the division
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = Variable(1, label = 'y')
        >>> v3 = v1 / v2
        >>> print(v3.val)
        3.0
        >>> print(v3.der)
        {'x': 1.0, 'y': -3.0}
        """
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
        """
        Calculates the division of two variables

        Parameters
        ----------
        other : Variable
            Variable to be divided
        
        Returns
        -------
        Variable
            Variable of the division
        
        Examples
        -------
        >>> v1 = Variable(1, label = 'x')
        >>> v2 = 3 / v1
        >>> print(v2.val)
        3.0
        >>> print(v2.der)
        {'x': -3.0}
        """
        new_var = self.__truediv__(other)
        # return reciprocal
        a2 = 1 / new_var.val
        dict2 = dict(new_var.der)
        for k in dict2:
            dict2[k] = (-1 * np.asarray(dict2[k])) / (new_var.val * new_var.val)
        reciprocal_new_var = Variable(a2, dict2, increment_counter=False)
        return reciprocal_new_var

    def __neg__(self):
        """
        Calculates the negative of a variable

        Returns
        -------
        Variable
            Variable of the negative
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = -v1
        >>> print(v2.val)
        -3.0
        >>> print(v2.der)
        {'x': -1.0}
        """
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
        """
        Calculates the power of a variable

        Parameters
        ----------
        other : Variable
            Variable to be powered
        
        Returns
        -------
        Variable
            Variable of the power
        
        Examples
        -------
        >>> v1 = Variable(3, label = 'x')
        >>> v2 = v1 ** 2
        >>> print(v2.val)
        9.0
        >>> print(v2.der)
        {'x': 6.0}
        """
        if isinstance(other, Variable):
            new_val = np.array([self.val ** other.val])
            new_der = {}
            all_ele = set(self.der.keys()) | set(other.der.keys())
            for v in list(all_ele): # v: element
                my_der = self.der[v] if v in self.der else 0
                other_der = other.der[v] if v in other.der else 0
                if v not in other.der:
                    new_der[v] = other.val * (self.val ** (other.val - 1)) * my_der
                else:
                    v_der = new_val * (other_der * np.log(self.val) + my_der / self.val * other.val)
                    #new_der[v] = np.array([v_der])
                    v_der[np.isnan(v_der)] = 0
                    new_der[v] = v_der
            return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
        else:
            try:
                new_val = self.val ** other
                new_der = {}
                for v in self.der:
                    #new_der[v] = [new_val[i] * self.der[v][i] * other / self.val[i] for i in range(len(new_val))]
                    new_der[v] = [other * (self.val[i] ** (other - 1)) * self.der[v][i] for i in range(len(new_val))]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
            except:
                #new_val = [self.val ** other]
                new_val = self.val ** other
                new_der = {}
                for v in self.der:
                    #new_der[v] = [new_val[0] * self.der[v] * other / self.val]
                    new_der[v] = self.der[v] * other * (self.val ** (other - 1))
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=True)


    def __rpow__(self, other):
        """
        Calculates the power of a variable

        Parameters
        ----------
        other : Variable
            Variable to be powered
        
        Returns
        -------
        Variable
            Variable of the power
        
        Examples
        -------
        >>> v1 = Variable(2, label = 'x')
        >>> v2 = np.e ** v1
        >>> print(v2.val)
        7.38905609893065
        >>> print(v2.der)
        {'x': 7.38905609893065}
        """
        if other == 0:
            new_der = {}
            for v in self.der:
                new_der[v] = 0
            return Variable(val=0, der=new_der, ad_mode=self.ad_mode, increment_counter=True)
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

    def __eq__(self, other):
        """
        Returns True if the two variables are equal

        Parameters
        ----------
        other : Variable
            Variable to be compared
        
        Returns
        -------
        bool
            True if the two variables are equal
        
        Examples
        -------
        >>> v1 = Variable(1)
        >>> v2 = Variable(1)
        >>> v3 = Variable(2)
        >>> v4 = Variable(1, {'x': 2})
        >>> print(v1 == v2)
        True
        >>> print(v1 == v3)
        False
        >>> print(v1 == v4)
        False
        """
        if not isinstance(other, Variable):
            return False
        if isinstance(self.val, (list, tuple, np.ndarray)) and isinstance(other.val, (list, tuple, np.ndarray)):
            if not np.all(self.val == other.val):
                return False
            if not compare_dicts_multi(self.der, other.der):
                return False
            return True

    def __ne__(self, other):
        """
        Returns True if the two variables are not equal

        Parameters
        ----------
        other : Variable
            Variable to be compared
        
        Returns
        -------
        bool
            True if the two variables are not equal
        
        Examples
        -------
        >>> v1 = Variable(1)
        >>> v2 = Variable(1)
        >>> v3 = Variable(2)
        >>> v4 = Variable(1, {'x': 2})
        >>> print(v1 != v2)
        False
        >>> print(v1 != v3)
        True
        >>> print(v1 != v4)
        True
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Returns True if the first variable is less than the second

        Parameters
        ----------
        other : Variable
            Variable to be compared
        
        Returns
        -------
        bool
            True if the first variable is less than the second
        
        Examples
        -------
        >>> v1 = Variable(1)
        >>> v2 = Variable(2)
        >>> print(v1 < v2)
        True
        """
        try:
            return self.val < other.val
        except:
            raise ValueError("Dimensions of the two variables are not equal")
    
    def __le__(self, other):
        """
        Returns True if the first variable is less than or equal to the second

        Parameters
        ----------
        other : Variable
            Variable to be compared
        
        Returns
        -------
        bool
            True if the first variable is less than or equal to the second
        
        Examples
        -------
        >>> v1 = Variable(1)
        >>> v2 = Variable(2)
        >>> print(v1 <= v2)
        True
        """
        try:
            return self.val <= other.val
        except:
            raise ValueError("Dimensions of the two variables are not equal")
    
    def __gt__(self, other):
        """
        Returns True if the first variable is greater than the second

        Parameters
        ----------
        other : Variable
            Variable to be compared
        
        Returns
        -------
        bool
            True if the first variable is greater than the second
        
        Examples
        -------
        >>> v1 = Variable(1)
        >>> v2 = Variable(2)
        >>> print(v1 > v2)
        False
        """
        try:
            return self.val > other.val
        except:
            raise ValueError("Dimensions of the two variables are not equal")
    
    def __ge__(self, other):
        """
        Returns True if the first variable is greater than or equal to the second

        Parameters
        ----------
        other : Variable
            Variable to be compared
        
        Returns
        -------
        bool
            True if the first variable is greater than or equal to the second
        
        Examples
        -------
        >>> v1 = Variable(2)
        >>> v2 = Variable(2)
        >>> print(v1 >= v2)
        True
        """
        try:
            return self.val >= other.val
        except:
            raise ValueError("Dimensions of the two variables are not equal")

def exp(x):
    """
    Calculates the exponential of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be exponentiated
    
    Returns
    -------
    Variable or float
        Exponential of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric exp(x)
    
    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = exp(v1)
    >>> print(v2.val)
    7.38905609893065
    >>> print(v2.der)
    {'x': 7.38905609893065}

    >>> v3 = exp(2)
    >>> print(v3)
    7.38905609893065
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
    Calculates the logarithm of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be logarithmized
    
    Returns
    -------
    Variable or float
        Logarithm of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric log10(x)
    
    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = log10(v1)
    >>> print(v2.val)
    0.301029995663981
    >>> print(v2.der)
    0.217147240951626

    >>> v3 = log10(2)
    >>> print(v3)
    0.301029995663981
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
    Calculates the natural logarithm of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be logarithmized
    
    Returns
    -------
    Variable or float
        Natural logarithm of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric ln(x)
    
    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = ln(v1)
    >>> print(v2.val)
    0.693147180559945
    >>> print(v2.der)
    {'x': 0.5}

    >>> v3 = ln(2)
    >>> print(v3)
    0.693147180559945
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
    Calculates the logistic function of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be logisticized
    
    Returns
    -------
    Variable or float
        Logistic of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric logistic(x)
    
    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = logistic(v1)
    >>> print(v2.val)
    0.880797077977882
    >>> print(v2.der)
    {'x': 0.104993585403506}

    >>> v3 = logistic(2)
    >>> print(v3)
    0.880797077977882
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

def sin(x):
    """
    Calculates the sine of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be sined
    
    Returns
    -------
    Variable or float
        Sine of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric sin(x)

    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = sin(v1)
    >>> print(v2.val)
    0.909297426825682
    >>> print(v2.der)
    {'x': -0.41614683654714}

    >>> v3 = sin(2)
    >>> print(v3)
    0.909297426825682
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

def cos(x):
    """
    Calculates the cosine of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be cosined
    
    Returns
    -------
    Variable or float
        Cosine of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric cos(x)
    
    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = cos(v1)
    >>> print(v2.val)
    -0.41614683654714
    >>> print(v2.der)
    {'x': -0.909297426825682}

    >>> v3 = cos(2)
    >>> print(v3)
    -0.41614683654714
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

def tan(x):
    """
    Calculates the tangent of a variable

    Parameters
    ----------
    x : Variable or float
        Variable or float to be tanged
    
    Returns
    -------
    Variable or float
        Tangent of the variable or float.
        If x is a Variable, returns a new variable with val and der
        If x is a number, returns numeric tan(x)
    
    Examples
    -------
    >>> v1 = Variable(2, label = 'x')
    >>> v2 = tan(v1)
    >>> print(v2.val)
    -2.185039863261519
    >>> print(v2.der)
    {'x': 5.774399204041917}

    >>> v3 = tan(2)
    >>> print(v3)
    -2.185039863261519
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
