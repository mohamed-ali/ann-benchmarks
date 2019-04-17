"""
Implements the necessary adapters to integrate LOPQ in the set
of benchmarked algorithms.
"""
from __future__ import absolute_import
import lopq
from ann_benchmarks.algorithms.base import BaseANN


class Lopq(BaseANN):
	"""Implements the methods required to integrate LOPQ
	to ann-benchmark framework"""
	def query(self):
		raise NotImplementedError

	def batch_query(self, X, n):
		raise NotImplementedError

	def get_batch_results(self):
		raise NotImplementedError

	def fit(self, X):
		raise NotImplementedError
