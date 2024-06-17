# Python
import numpy as np
import argparse
import os

# Torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# Haven
from haven import haven_utils as hu
from haven import haven_wizard as hw

# Pandas
import pandas as pd

# DeepFish
import exp_configs
from DeepFish import utils as ut
from DeepFish import datasets, models
from DeepFish import wrappers

# -----------------------------------------------------------------------------
def trainval(exp_dict, savedir, args):
	"""
	exp_dict: dictionary defining the hyperparameters of the experiment
	savedir: the directory where the experiment will be saved
	args: arguments passed through the command line
	"""

	# Set seed
	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)
	
	# Select device
	if args.use_cuda:
		device = "cuda"
		torch.cuda.manual_seed_all(seed)
		assert torch.cuda.is_available(), "cuda is not, available please run with \"-uc 0\""
	else:
		device = "cpu"

	print("Running on device: %s" % device)
	
	# Dataset - train, validation and visualization
	train_loader, val_loader, vis_loader = loadData(exp_dict)

	# Create model and optimizer
	model, opt = createModelAndOpt(exp_dict)

	score_list = []

	# Checkpointing
	score_list_path = os.path.join(savedir, "score_list.pkl")
	model_path = os.path.join(savedir, "model_state_dict.pth")
	opt_path = os.path.join(savedir, "opt_state_dict.pth")

	if os.path.exists(score_list_path):
		# Resume experiment
		score_list = hu.load_pkl(score_list_path)
		model.load_state_dict(torch.load(model_path))
		opt.load_state_dict(torch.load(opt_path))
		s_epoch = score_list[-1]["epoch"] + 1
		print(f"Resume experiment from epoch {s_epoch}.")
	else:
		# Restart experiment
		score_list = []
		s_epoch = 0
		print("No score_list found. Start or restart experiment.")

	# Run training and validation
	for epoch in range(s_epoch, exp_dict["max_epoch"]):
		score_dict = {"epoch": epoch}

		# Visualize
		model.vis_on_loader(vis_loader, savedir=os.path.join(savedir, "images"))
		
		# Train
		score_dict.update(model.train_on_loader(train_loader))
		
		# Validate
		score_dict.update(model.val_on_loader(val_loader))

		# Add score_dict to score_list
		score_list += [score_dict]

		# Report and save
		print(pd.DataFrame(score_list).tail())
		hu.save_pkl(score_list_path, score_list)
		hu.torch_save(model_path, model.state_dict())
		hu.torch_save(opt_path, opt.state_dict())
		print("Saved in {savedir}\n")
	
# -----------------------------------------------------------------------------
def loadData(exp_dict):
	try:
		# Get train and validation datasets
		datadir = args.datadir
		train_set = datasets.get_dataset("train", exp_dict, datadir)
		val_set = datasets.get_dataset("val", exp_dict, datadir)
		
		# Create train, validation and visualization DataLoader
		batch_size = exp_dict["batch_size"]
		train_num_samples = max(min(500, len(train_set)), len(val_set))
		random_sampler = RandomSampler(train_set, True, train_num_samples)

		train_loader = DataLoader(train_set, batch_size, sampler=random_sampler)
		val_loader = DataLoader(val_set, batch_size, shuffle=False)
		vis_loader = DataLoader(val_set, 1, sampler=ut.SubsetSampler(train_set, indices=[0, 1, 2]))
	except Exception as ex:
		print(ex)
		exit()

	return train_loader, val_loader, vis_loader

# -----------------------------------------------------------------------------
def createModelAndOpt(exp_dict):
	model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
	opt = torch.optim.Adam(model_original.parameters(), lr=1e-5, weight_decay=0.0005)

	model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()

	return model, opt

# -----------------------------------------------------------------------------
if __name__ == "__main__":
	import exp_configs

	parser = argparse.ArgumentParser()

	parser.add_argument("-e", "--exp_group_list", nargs="+",
						help="Define which exp groups to run.")
	parser.add_argument("-sb", "--savedir_base", default=None,
						help="Define the base directory where the experiments will be saved.")
	parser.add_argument("-d", "--datadir", default=None,
						help="Define the dataset directory.")
	parser.add_argument("-r", "--reset",  default=0, type=int,
						help="Reset or resume the experiment.")
	parser.add_argument("--debug",  default=False, type=int,
						help="Debug mode.")
	parser.add_argument("-ei", "--exp_id", default=None,
						help="Run a specific experiment based on its id.")
	parser.add_argument("-j", "--run_jobs", default=0, type=int,
						help="Run the experiments as jobs in the cluster.")
	parser.add_argument("-nw", "--num_workers", type=int, default=0,
						help="Specify the number of workers in the dataloader.")
	parser.add_argument("-v", "--visualize_notebook", type=str, default="",
						help="Create a jupyter file to visualize the results.")
	parser.add_argument("-uc", "--use_cuda", type=int, default=1)

	args, others = parser.parse_known_args()

	# Launch experiments using magic command
	hw.run_wizard(func=trainval, exp_groups=exp_configs.EXP_GROUPS, args=args)
