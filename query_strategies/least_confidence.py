import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		U = probs.max(1)[0]
		#A.max(0)：返回A每一列最大值组成的一维数组；
		#A.max(1)：返回A每一行最大值组成的一维数组；
		#然后将这些各个最大置信度里面 挑选其中最小的 选出600个最不自信的 变为待标记集合
		return idxs_unlabeled[U.sort()[1][:n]]
