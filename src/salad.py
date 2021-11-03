class Variable(object):
	counter = 0
	def __init__(self, val, label=None, der=None, ad_mode='forward'):

		self.label = 'v' + Variable.get_counter()

		if der is not None:
			self.der = der
		else:
			self.der = {label: [[1]]}

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


x = Variable(val=2)
y = Variable(val=3)
z = Variable(val=4)

print(x)
print(y)
print(z)