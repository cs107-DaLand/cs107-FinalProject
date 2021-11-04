import numpy as np
from utils import add_dict

class Variable(object):
	counter = 0
	def __init__(self, val, der=None, label=None, ad_mode='forward'):
		
		self.val = np.asarray(val)
		self.ad_mode = ad_mode

		if label is not None:
			self.label = label
		else:
			self.label = 'v' + Variable.get_counter()

		if der is not None:
			self.der = der
		else:
			self.der = {self.label: np.ones(self.val.shape)}

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
			for v in set(dict1) - set(dict2): # Only those variables in self
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
		pass

	def __rtruediv__(self, other):
		pass