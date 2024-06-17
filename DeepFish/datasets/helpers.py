# TorchVision
from torchvision import transforms

# -----------------------------------------------------------------------------
def slice_df(df, habitat):
	if habitat is None:
		return df
	
	return df[df['ID'].apply(lambda x: True if x.split("/")[0] == habitat else False)]

# -----------------------------------------------------------------------------
def slice_df_reg(df, habitat):
	if habitat is None:
		return df
	
	return df[df['ID'].apply(lambda x: True if x.split("/")[1].split("_")[0] == habitat else False)]

# -----------------------------------------------------------------------------
def get_transformer(transform):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	normalize_transform = transforms.Normalize(mean, std)

	if transform == "resize_normalize":
		return transforms.Compose([transforms.Resize((224,224)),
									transforms.ToTensor(),
									normalize_transform])

	if transform == "rgb_normalize":
		return transforms.Compose([transforms.ToTensor(),
							 		normalize_transform])
