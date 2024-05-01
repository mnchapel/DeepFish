# [MN] Seems not used

# # Python
# import numpy as np

# # Torch
# import torch
# import torch.nn as nn
# from torchvision import models

# # Haven
# from haven import haven_utils
# from haven import haven_img

# # SciPy
# import scipy

# # Numba
# import numba

# # OpenCV
# import cv2

# # DeephFish
# from src import utils as ut

# ###############################################################################
# class CSRNet(nn.Module):
	
# 	# -------------------------------------------------------------------------
# 	def __init__(self, ):
# 		super().__init__()
# 		self.seen = 0
# 		self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
# 		self.backend_feat = [512, 512, 512, 256, 128, 64]
# 		self.frontend = make_layers(self.frontend_feat)
# 		self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
# 		self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

# 		mod = models.vgg16(pretrained=True)
# 		self._initialize_weights()
# 		for i in range(len(self.frontend.state_dict().items())):
# 			items1 = list(self.frontend.state_dict().items())
# 			items2 = list(mod.state_dict().items())
# 			items1[i][1].data[:] = items2[i][1].data[:]

# 		self.opt = torch.optim.SGD(self.parameters(),
# 									1e-7,
# 									momentum=0.95,
# 									weight_decay=5*1e-4)

# 	# -------------------------------------------------------------------------
# 	def train_step(self, batch):
# 		self.train()

# 		images = batch["images"].cuda()

# 		self.opt.zero_grad()
# 		points = batch["points"].long().cuda()
# 		points_numpy = haven_utils.t2n(points).squeeze()
# 		target = get_density(points_numpy)
# 		target = cv2.resize(target,
# 							(target.shape[1] // 8,
# 							 target.shape[0] // 8),
# 							interpolation=cv2.INTER_CUBIC) * 64
# 		output = self(images)

# 		self.train()
# 		criterion = nn.MSELoss(size_average=False).cuda()
# 		loss = criterion(output.squeeze(),
# 						 torch.FloatTensor(target).cuda())
# 		loss.backward()
# 		self.opt.step()

# 		return loss.item()

# 	# -------------------------------------------------------------------------
# 	# @torch.no_grad()
# 	# def visualize(self, batch, **options):
# 	#     self.eval()
# 	#     self.predict(batch, method="blobs")
# 	@torch.no_grad()
# 	def visualize(self, batch, savedir, **options):
# 		index = batch["meta"]["index"].item()

# 		path_base = "%s/%d" % (savedir, index)

# 		counts = self.predict(batch, method="counts")
# 		points = self.predict(batch, method="points")
		
# 		name = "pred_points"
# 		img1 = haven_img.get_image(np.array(batch["images_original"]) * 0.4 + 0.6 * haven_img.t2n(haven_img.gray2cmap(points)))
# 		meta_dict = {"index": index, "name": name}
# 		ut.save_img_pkl(img1, meta_dict, path_base, name=name)

# 		name = "gt_points"
# 		img1 = haven_img.get_image(np.array(batch["images_original"]), mask=batch["points"].long().squeeze(), enlarge=1)
# 		meta_dict = {"pred": counts[0], "counts": batch["counts"].item(), "name":name}
# 		ut.save_img_pkl(img1, meta_dict, path_base, name=name)

# 	# -------------------------------------------------------------------------
# 	@torch.no_grad()
# 	def predict(self, batch, **options):
# 		self.eval()
		
# 		if options["method"] == "counts":
# 			images = batch["images"].cuda()
# 			output = self(images).sum().item()
			
# 			return np.asarray([output], dtype=np.float32)

# 		elif options["method"] == "points":
# 			images = batch["images"].cuda()
# 			output = self(images).cpu().numpy().squeeze()
# 			n,c,h,w = images.shape
# 			preds = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC) / 64

# 			return preds

# 	# -------------------------------------------------------------------------
# 	def forward(self, x):
# 		x = self.frontend(x)
# 		x = self.backend(x)
# 		x = self.output_layer(x)
# 		return x

# 	# -------------------------------------------------------------------------
# 	def _initialize_weights(self):
# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d):
# 				nn.init.normal_(m.weight, std=0.01)
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0)
# 			elif isinstance(m, nn.BatchNorm2d):
# 				nn.init.constant_(m.weight, 1)
# 				nn.init.constant_(m.bias, 0)


# # -----------------------------------------------------------------------------
# def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
# 	if dilation:
# 		d_rate = 2
# 	else:
# 		d_rate = 1
# 	layers = []
# 	for v in cfg:
# 		if v == 'M':
# 			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 		else:
# 			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
# 			if batch_norm:
# 				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
# 			else:
# 				layers += [conv2d, nn.ReLU(inplace=True)]
# 			in_channels = v
# 	return nn.Sequential(*layers)

# # -----------------------------------------------------------------------------
# @numba.jit()
# def get_density(gt):
# 	density = np.zeros(gt.shape, dtype=np.float32)
# 	gt_count = np.count_nonzero(gt)
# 	if gt_count == 0:
# 		return density

# 	pts = np.array(list(zip(np.nonzero(gt)[1],
# 							np.nonzero(gt)[0])))
# 	leafsize = 2048
# 	# build kdtree
# 	tree = scipy.spatial.KDTree(pts.copy(),
# 								leafsize=leafsize)
# 	# query kdtree
# 	distances, locations = tree.query(pts, k=4)

# 	for i, pt in enumerate(pts):
# 		pt2d = np.zeros(gt.shape, dtype=np.float32)
# 		pt2d[pt[1], pt[0]] = 1.
# 		if gt_count > 1:
# 			sigma = (distances[i][1] + distances[i][2]
# 					 + distances[i][3]) * 0.1
# 		else:
# 			sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
# 		density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

# 	return density
