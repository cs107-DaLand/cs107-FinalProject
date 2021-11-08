import numpy as np
from utils import *


class Variable(object):
    counter = 0
    def __init__(self, val, der=None, label=None, ad_mode='forward', increment_counter=True):
        self.val = np.asarray(val)
        self.ad_mode = ad_mode
        self.increment_counter = increment_counter

        if label is not None:
            self.label = label
        else:
            self.label = 'v' + Variable.get_counter()

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
        return self.__add__(-1*other)

    def __rsub__(self, other):
        return (-1*self).__add__(other)

    def __mul__(self, other):
        try:
            a1, a2 = self.val, other.val
            dict1, dict2 = self.der, other.der
            new_dict = {}
            for v in set(dict1).intersection(set(dict2)): #intersection
                new_dict[v] = a1*dict2[v] + a2*dict1[v]
            for v in set(dict1) - set(dict2): # Only those variables in self DOES THIS WORK
                new_dict[v] = a2*dict1[v]
            for v in set(dict2) - set(dict1): # Only those variables in other
                new_dict[v] = a1*dict2[v]
            return Variable(a1*a2, new_dict)

        except AttributeError:
            new_dict = {}
            for key in self.der:		
                new_dict[key] = other*self.der[key]	
            return Variable(self.val * other, new_dict)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            # divide other by 1
            a2 = 1/other.val
            dict2 = dict(other.der)
            for k in dict2:
                dict2[k] = (-1*dict2[k]) / (other.val * other.val)
            reciprocal_other = Variable(a2, dict2, increment_counter=False)
            return self.__mul__(reciprocal_other)

        except AttributeError:
            # divide self by other
            a1 = self.val/other
            dict1 = dict(self.der)
            for k in dict1:
                dict1[k] = (other*dict1[k]) / (other*other)
            return Variable(a1, dict1)


    def __rtruediv__(self, other):
        new_var = self.__truediv__(other)
        # return reciprocal
        a2 = 1/new_var.val
        dict2 = dict(new_var.der)
        for k in dict2:
            dict2[k] = (-1*dict2[k]) / (new_var.val * new_var.val)
        reciprocal_new_var = Variable(a2, dict2, increment_counter=False)
        return reciprocal_new_var

    def __neg__(self):
        new_dir = {}
        try:
            for key in self.der:
                new_dir[key] = [-i for i in self.der[key]]
            return Variable(val=-self.val, der=new_dir, ad_mode=self.ad_mode, increment_counter=self.increment_counter)
        except:
            for key in self.der:
                new_dir[key] = [-self.der[key]]
            return Variable(val=[-self.val], der=new_dir, ad_mode=self.ad_mode, increment_counter=self.increment_counter)

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
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)
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
                    new_der[v] = np.array([v_der])
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)
        else:
            try:
                new_val = self.val ** other
                new_der = {}
                for v in self.der:
                    new_der[v] = [new_val[i] * self.der[v][i] * other / self.val[i] for i in range(len(new_val))]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)
            except:
                new_val = [self.val ** other]
                new_der = {}
                for v in self.der:
                    new_der[v] = [new_val[0] * self.der[v] * other / self.val]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)

    def __rpow__(self, other):
        if isinstance(other, Variable):
            return other ** self
        else:
            try:
                new_val = other ** self.val
                new_der = {}
                for v in self.der:
                    new_der[v] = [new_val[i] * self.der[v][i] * np.log(other) for i in range(len(new_val))]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)
            except:
                new_val = [other ** self.val]
                new_der = {}
                for v in self.der:
                    new_der[v] = [new_val[0] * self.der[v] * np.log(other)]
                return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)


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
