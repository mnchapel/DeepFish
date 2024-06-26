# Python
import numpy as np

###############################################################################
class LocMonitor:
	
	# -------------------------------------------------------------------------
	def __init__(self):
		self.ae = 0
		self.n_samples = 0

	# -------------------------------------------------------------------------
	def add(self, ae):
		self.ae += ae.sum()
		self.n_samples += ae.shape[0]

	# -------------------------------------------------------------------------
	def get_avg_score(self):
		return self.ae/ self.n_samples

###############################################################################
class SegMonitor:
	
	# -------------------------------------------------------------------------
	def __init__(self):
		self.cf = None

	# -------------------------------------------------------------------------
	def add(self, cf):
		if self.cf is None:
			self.cf = cf 
		else:
			self.cf += cf

	# -------------------------------------------------------------------------
	def get_avg_score(self):
		Inter = np.diag(self.cf)
		G = self.cf.sum(axis=1)
		P = self.cf.sum(axis=0)
		union = G + P - Inter

		nz = union != 0
		mIoU = Inter[nz] / union[nz]
		mIoU = np.mean(mIoU)

		return 1. - mIoU
