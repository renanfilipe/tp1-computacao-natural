import random


class NotRandom():
	def __init__(self, seed: int = 1000):
		self.seed = seed

	def _call(self, name: str, *args, **kwargs):
		self.seed += 1
		func = getattr(random, name)
		random.seed(self.seed)
		result = func(*args, **kwargs)
		# print(self.seed, name, result)
		return result

	def uniform(self, *args, **kwargs):
		return self._call("uniform", *args, **kwargs)

	def randrange(self, *args, **kwargs):
		return self._call("randrange", *args, **kwargs)

	def choices(self, *args, **kwargs):
		return self._call("choices", *args, **kwargs)

	def choice(self, *args, **kwargs):
		return self._call("choice", *args, **kwargs)

	def sample(self, *args, **kwargs):
		return self._call("sample", *args, **kwargs)
