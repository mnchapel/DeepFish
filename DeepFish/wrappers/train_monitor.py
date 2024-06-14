###############################################################################
class TrainMonitor:
	
	# -------------------------------------------------------------------------
	def __init__(self):
		self.score_dict_sum = {}
		self.n = 0

	# -------------------------------------------------------------------------
	def add(self, score_dict):
		for k,v in score_dict.items():
			if k not in self.score_dict_sum:
				self.score_dict_sum[k] = score_dict[k]
			else:
				self.n += 1
				self.score_dict_sum[k] += score_dict[k]

	# -------------------------------------------------------------------------
	def get_avg_score(self):
		return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}
