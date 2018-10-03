import numpy


class NotRandom:
	def __init__(self, seed: int = 10000):
		self.seed = seed

	def update_seed(self):
		self.seed += 10000

	def _call(self, name: str, *args, **kwargs):
		self.seed += 1
		func = getattr(numpy.random, name)
		numpy.random.seed(self.seed)
		result = func(*args, **kwargs)
		# print(self.seed, name, result)
		return result

	def uniform(self, *args, **kwargs):
		return self._call("uniform", *args, **kwargs)

	def randrange(self, *args, **kwargs):
		return self._call("randint", *args, **kwargs)

	def choice(self, *args, **kwargs):
		return self._call("choice", *args, **kwargs)

	def sample(self, *args, **kwargs):
		return self._call("sample", *args, **kwargs)
