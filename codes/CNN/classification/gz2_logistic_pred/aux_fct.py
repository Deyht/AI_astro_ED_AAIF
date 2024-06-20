
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from PIL import Image
from PIL import ImageFilter

import albumentations as A
import cv2
import os,sys
import gc


raw_image_size = 424
image_size = 64
im_depth = 3


if(not os.path.isdir("gz2_images_%dx%d"%(image_size,image_size))):
	print("Downloading dataset ...")
	if(image_size == 64):
		os.system("wget https://share.obspm.fr/s/HJAXeYkiBsK4P8F/download/gz2_images_64x64.tar.gz")
		os.system("tar -xzf gz2_images_64x64.tar.gz")
	elif(image_size == 128):
		os.system("wget https://share.obspm.fr/s/fGRRQX6zao43mbE/download/gz2_images_128x128.tar.gz")
		os.system("tar -xzf gz2_images_128x128.tar.gz")
	else:
		print("Invalid image size ...")
		exit()

catalog_path = "gz2_filename_mapping_with_all_flags.csv"

if(not os.path.isfile(catalog_path)):
		os.system("wget https://share.obspm.fr/s/2GPmEDzdCz8Jq9H/download/gz2_filename_mapping_with_all_flags.csv")

gz2_catalog_header = np.genfromtxt(catalog_path, delimiter=",", max_rows=1, dtype="str")

master_classes, class_index, nb_per_group = np.unique(gz2_catalog_header[4:].astype("<U3"), return_inverse=True, return_counts=True)

print (master_classes, nb_per_group)


gz2_catalog = np.genfromtxt(catalog_path, delimiter=",", skip_header=1, dtype="float")

do_not_exist = np.zeros(np.shape(gz2_catalog), dtype="int")

for i in range(0, np.shape(gz2_catalog)[0]):
	path = "gz2_images_%dx%d/%d.jpg"%(image_size, image_size, int(gz2_catalog[i,1]))
	if(not os.path.isfile(path)):
		do_not_exist[i] = 1

#Remove around 100 objects for which the image is not provided
index = np.where(do_not_exist == 1)[0]

gz2_catalog = np.delete(gz2_catalog, index, axis=0)

np.random.shuffle(gz2_catalog)

for master_class in master_classes:
	subset_index = np.where(gz2_catalog_header[4:].astype("<U3") == master_class)[0]
	cat_subset = gz2_catalog[:,4+subset_index]
	
	index_do_not_apply = np.where(np.sum(cat_subset, axis=1) < 0.01)[0]
	if(len(index_do_not_apply) > 0):
		gz2_catalog_header = np.append(gz2_catalog_header, master_class)
		new_column = np.zeros(np.shape(gz2_catalog)[0])
		new_column[index_do_not_apply] = 1.0
		gz2_catalog = np.column_stack((gz2_catalog, new_column))
	
		subset_index = np.where(gz2_catalog_header[4:].astype("<U3") == master_class)[0]
		cat_subset = gz2_catalog[:,4+subset_index]
	
	arg_max_array = np.argmax(cat_subset, axis=1)
	gz2_catalog[:,4+subset_index] = 0.0
	pos_max = subset_index[arg_max_array]
	for k in range(0,len(subset_index)):
		index = np.where(pos_max == subset_index[k])[0]
		gz2_catalog[index,4+subset_index[k]] = 1.0


total_nb_images = np.shape(gz2_catalog)[0]

nb_class = len(gz2_catalog_header[4:])

frac_test = 0.05

nb_test = int(frac_test*total_nb_images)
nb_train = total_nb_images - nb_test

gz2_test = gz2_catalog[:nb_test]
gz2_train = gz2_catalog[nb_test:]


crop_size = int(np.maximum(raw_image_size//2,image_size))

transform = A.Compose([
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
])

transform_val = A.Compose([
	A.CenterCrop(crop_size, crop_size),
	A.LongestMaxSize(max_size=image_size, interpolation=1),
])

nb_im_train = 2048
nb_im_test = np.shape(gz2_test)[0]

input_data = np.zeros((nb_im_train,image_size*image_size*(im_depth+1)), dtype="float32")
targets = np.zeros((nb_im_train,nb_class), dtype="float32")

input_test = np.zeros((nb_im_test,image_size*image_size*(im_depth+1)), dtype="float32")
targets_test = np.zeros((nb_im_test,nb_class), dtype="float32")
	
zero_target = np.zeros(nb_class)

def create_train_batch(visual=0):
	
	if(visual):
		fig, axs = plt.subplots(4,5, figsize=(5,4), dpi=250, constrained_layout=True)

	for i in range(0,nb_im_train):
		i_d = int(np.random.random()*nb_train)
		path = "gz2_images_%dx%d/%d.jpg"%(image_size, image_size, int(gz2_train[i_d,1]))
		
		patch = np.asarray(Image.open(path))
		transformed = transform(image=patch)
		patch = (transformed['image']/255.0)
		
		for depth in range(0,im_depth):
			input_data[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = (np.copy(patch[:,:,depth]).flatten("C"))
		input_data[i,3*image_size*image_size:] = 0.0
		targets[i,:] = gz2_train[i_d,4:]

		if(visual and i < 20):
			axs[i//5][i%5].imshow(patch)
			axs[i//5][i%5].set_axis_off()

	if(visual):
		plt.savefig("training_set_example.jpg", dpi=250)
		return

	return input_data, targets


def create_test_batch():
	
	for i in range(0,nb_im_test):
		path = "gz2_images_%dx%d/%d.jpg"%(image_size,image_size, int(gz2_test[i,1]))
		
		patch = np.asarray(Image.open(path))/255.0
	
		for depth in range(0,im_depth):
			input_test[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = (np.copy(patch[:,:,depth]).flatten("C"))
		input_test[i,3*image_size*image_size:] = 0.0
		targets_test[i,:] = 0
		targets_test[i,:] = gz2_test[i,4:]

	return input_test, targets_test


