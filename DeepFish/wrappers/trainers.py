# Python
import tqdm
import os

# Torch
import torch

# DeepFish
from .train_monitor import TrainMonitor

# -----------------------------------------------------------------------------
def train_on_loader(model, train_loader):
	model.train()

	train_monitor = TrainMonitor()

	print("Training")
	
	for batch in tqdm.tqdm(train_loader):
		score_dict = model.train_on_batch(batch)
		
		train_monitor.add(score_dict)
		
	return train_monitor.get_avg_score()

# -----------------------------------------------------------------------------
@torch.no_grad()
def val_on_loader(model, val_loader, val_monitor):
	model.eval()

	print("Validating")
	
	for batch in tqdm.tqdm(val_loader):
		score = model.val_on_batch(batch)
		val_monitor.add(score)

	return val_monitor.get_avg_score()

# -----------------------------------------------------------------------------
@torch.no_grad()
def vis_on_loader(model, vis_loader, savedir):
	model.eval()
	
	for i, batch in enumerate(vis_loader):
		print("%d - visualizing %s image - savedir: %s" % (i, batch["meta"]["split"][0], savedir.split("\\")[-2])) # Warning: on windows \ and not /
		model.vis_on_batch(batch, savedir_image=os.path.join(savedir, f"{i}.png"))
		
# -----------------------------------------------------------------------------
@torch.no_grad()
def test_on_loader(model, test_loader):
	model.eval()
	ae = 0.
	n_samples = 0.

	n_batches = len(test_loader)
	pbar = tqdm.tqdm(total=n_batches)
	
	for batch in test_loader:
		pred_count = model.predict(batch, method="counts")

		ae += abs(batch["counts"].cpu().numpy().ravel() - pred_count.ravel()).sum()
		n_samples += batch["counts"].shape[0]

		pbar.set_description("TEST mae: %.4f" % (ae / n_samples))
		pbar.update(1)

	pbar.close()
	score = ae / n_samples
	print({"test_score": score, "test_mae": score})

	return {"test_score": score, "test_mae": score}
