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
class SegWrapper(torch.nn.Module):
	
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
		val_monitor = SegMonitor()
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
		mask_classes = batch["mask_classes"].cuda()

		# Forward + loss
		logits = self.model.forward(images)
		p_log = F.log_softmax(logits, dim=1)
		p = F.softmax(logits, dim=1)
		FL = p_log*(1.-p)**2.
		loss = F.nll_loss(FL, mask_classes.long())

		# Backward + optimizer
		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		return {"loss_seg": loss.item()}

	# -------------------------------------------------------------------------
	def val_on_batch(self, batch, **extras):
		pred_seg = self.predict_on_batch(batch)
		cm_pytorch = confusion(torch.from_numpy(pred_seg).cuda().float(), batch["mask_classes"].cuda().float())
			
		return cm_pytorch

	# -------------------------------------------------------------------------
	def vis_on_batch(self, batch, savedir_image):
		from skimage.segmentation import mark_boundaries
		from skimage import color
		from skimage.measure import label
		
		pred_mask = self.predict_on_batch(batch)

		img = hu.get_image(batch["images"], denorm="rgb")
		img_np = np.array(img)
		pm = pred_mask.squeeze()
		out = color.label2rgb(label(pm), image=(img_np), image_alpha=1.0, bg_label=0)
		img_mask = mark_boundaries(out.squeeze(),  label(pm).squeeze())
		out = color.label2rgb(label(batch["mask_classes"][0]), image=(img_np), image_alpha=1.0, bg_label=0)
		img_gt = mark_boundaries(out.squeeze(),  label(batch["mask_classes"]).squeeze())
		hu.save_image(savedir_image, np.hstack([img_gt, img_mask]))

	# -------------------------------------------------------------------------
	def predict_on_batch(self, batch):
		self.eval()
		images = batch["images"].cuda()
		pred_mask = self.model.forward(images).data.max(1)[1].squeeze().cpu().numpy()

		return pred_mask[None]
					
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

		return {"val_seg":1. - mIoU}

# -----------------------------------------------------------------------------
# seg
def confusion(prediction, truth):
	confusion_vector = prediction / truth

	tp = torch.sum(confusion_vector == 1).item()
	fp = torch.sum(confusion_vector == float('inf')).item()
	tn = torch.sum(torch.isnan(confusion_vector)).item()
	fn = torch.sum(confusion_vector == 0).item()
	cm = np.array([[tn,fp],[fn,tp]])
	
	return cm
