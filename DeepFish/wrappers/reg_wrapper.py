# Python
import numpy as np

# Torch
import torch
import torch.nn.functional as F

# Haven
from haven import haven_utils as hu

# DeepFish
from .trainers import train_on_loader, val_on_loader, vis_on_loader

###############################################################################
class RegWrapper(torch.nn.Module):
	
	# -------------------------------------------------------------------------
	def __init__(self, model, opt):
		super().__init__()
		self.model = model
		self.opt = opt

	# -------------------------------------------------------------------------
	#                                   On Loader
	# -------------------------------------------------------------------------

	# -------------------------------------------------------------------------
	def train_on_loader(self, train_loader):
		return train_on_loader(self, train_loader)

	# -------------------------------------------------------------------------
	def val_on_loader(self, val_loader):
		val_monitor = RegMonitor()
		return val_on_loader(self, val_loader, val_monitor=val_monitor)

	# -------------------------------------------------------------------------
	def vis_on_loader(self, vis_loader, savedir):
		return vis_on_loader(self, vis_loader, savedir=savedir)

	# -------------------------------------------------------------------------
	#                                   On Bacth
	# -------------------------------------------------------------------------

	# -------------------------------------------------------------------------
	def train_on_batch(self, batch, **extras):
		# Data
		images = batch["images"].cuda()
		counts = batch["counts"].cuda()

		# Forward + loss
		pred_counts = self.model.forward(images)
		loss_reg = F.mse_loss(pred_counts.squeeze(), counts.float().squeeze())

		# Backward + optimizer
		self.opt.zero_grad()
		loss_reg.backward()
		self.opt.step()

		return {"loss_reg": loss_reg.item()}

	# -------------------------------------------------------------------------
	def val_on_batch(self, batch, **extras):
		preds = self.predict_on_batch(batch)
		val_reg = abs(preds.detach().cpu().numpy().ravel() - batch["counts"].numpy().ravel())

		return val_reg
		
	# -------------------------------------------------------------------------
	def vis_on_batch(self, batch, savedir_image):		
		pred_counts = self.predict_on_batch(batch)
		img = hu.get_image(batch["image_original"], denorm="rgb")
		img = np.array(img)
		hu.save_image(savedir_image+"/images/%d.jpg" % batch["meta"]["index"], img)
		hu.save_json(savedir_image+"/images/%d.json" % batch["meta"]["index"],
					{"pred_counts":float(pred_counts), "gt_counts": float(batch["counts"])})

	# -------------------------------------------------------------------------
	def predict_on_batch(self, batch):
		images = batch["images"].cuda()
		
		return self.model.forward(images).round()

###############################################################################
class RegMonitor:
	
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
		return {"val_reg":self.ae/ self.n_samples}
		