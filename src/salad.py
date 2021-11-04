import numpy as np
from utils import add_dict

class Variable(object):
	counter = 0
	def __init__(self, val, der=None, label=None, ad_mode='forward'):

		if label is not None:
			self.label = label
		else:
			self.label = 'v' + Variable.get_counter()

		if der is not None:
			self.der = der
		else:
			self.der = {label: np.asarray([[1]])}

		self.ad_mode = ad_mode
		self.val = val

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
		pass
		return self.__add__(other)

	def __sub__(self, other):
		pass

	def __rsub__(self, other):
		pass

	def __mul__(self, other):
		pass

	def __rmul__(self, other):
		pass

	def __truediv__(self, other):
		pass

	def __rtruediv__(self, other):
		pass
