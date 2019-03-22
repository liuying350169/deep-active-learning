import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(MarginSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		print(self.idxs_lb)
		print(idxs_unlabeled)

		#~按位取反  即 未标记集合是已标记集合的子集
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		#利用当前的模型对未标记样本池中的样本进行预测
		probs_sorted, idxs = probs.sort(descending=True)

		# for i in range(len(probs_sorted)):
		# 	print(probs_sorted[i])
		# 	print(idxs[i])

		#print(probs_sorted[0:1, 2])
		#对于预测的标签排序 ， 从高到低依次是属于各个类别的概率
		U = probs_sorted[:, 0] - probs_sorted[:, 1]
		#用前面的  对应每个位  减去后面的

		# print(len(probs_sorted[:, 0]))
		# print(probs_sorted[:, 1])
		# print(U)

		#表示对一个二维数组 取第一维的所有数据和第二维的所有数据
		#每一维的个数为 整个未标记样本的个数
		#也就是用每个未标记样本的个数的 最大概率  减去  第二大的概率
		#这就是现成的BvSB

		#求得待标记集
		#对边缘最小的n个 取为待标记，n为每轮标记的个数

		return idxs_unlabeled[U.sort()[1][:n]]
