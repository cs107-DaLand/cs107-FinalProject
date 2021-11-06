import numpy as np

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

    def __neg__(self):
        new_dir = {}
        for key in self.der:
            new_dir[key] = [-num for num in self.der[key]]
        return Variable(val=-self.val, der=new_dir, ad_mode=self.ad_mode, increment_counter=self.increment_counter)

    def __pow__(self, other):
        if isinstance(other, Variable):
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
        else:
            new_val = self.val ** other
            new_der = {}
            for v in self.der:
                new_der[v] = [new_val[i] * self.der[v][i] * other / self.val[i] for i in range(len(new_val))]
            return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)

    def __rpow__(self, other):
        if isinstance(other, Variable):
            return other ** self
        else:
            new_val = other ** self.val
            new_der = {}
            for v in self.der:
                new_der[v] = [new_val[i] * self.der[v][i] * np.log(other) for i in range(len(new_val))]
            return Variable(val=new_val, der=new_der, ad_mode=self.ad_mode, increment_counter=self.increment_counter)


    def __ne__(self, other):
        return self.label != other.label
    
    def __lt__(self, other):
        if not len(self.val) == len(other.val):
            raise ValueError("Dimensions of the two variables are not equal")
        return self.val < other.val
    
    def __le__(self, other):
        if not len(self.val) == len(other.val):
            raise ValueError("Dimensions of the two variables are not equal")
        return self.val <= other.val
    
    def __gt__(self, other):
        if not len(self.val) == len(other.val):
            raise ValueError("Dimensions of the two variables are not equal")
        return self.val > other.val
    
    def __ge__(self, other):
        if not len(self.val) == len(other.val):
            raise ValueError("Dimensions of the two variables are not equal")
        return self.val >= other.val

# test cases
x = Variable(val=np.array([1,2,3]), label='x')
v = Variable(val=[3,4,5])
v.der = {'x': np.array([1,2,3]), 'y': np.array([7,8,9])}
print(-v)
print(v<x)
print(np.e**(x**2))
print(x**v)
print(v**x)