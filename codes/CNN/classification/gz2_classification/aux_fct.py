
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

catalog_path = "gz2_filename_mapping_with_class.csv"

if(not os.path.isfile(catalog_path)):
		os.system("wget https://share.obspm.fr/s/fKqtaAQTQAt7k7J/download/gz2_filename_mapping_with_class.csv")


gz2_catalog_header = np.genfromtxt(catalog_path, delimiter=",", max_rows=1, dtype="str")

gz2_catalog = np.genfromtxt(catalog_path, delimiter=",", skip_header=1, dtype="str")

np.random.shuffle(gz2_catalog)

do_not_exist = np.zeros(np.shape(gz2_catalog), dtype="int")

for i in range(0, np.shape(gz2_catalog)[0]):
	path = "gz2_images_%dx%d/%s.jpg"%(image_size, image_size, gz2_catalog[i,1])
	if(not os.path.isfile(path)):
		do_not_exist[i] = 1

#Remove around 100 objects for which the image is not provided
index = np.where(do_not_exist == 1)[0]

gz2_catalog = np.delete(gz2_catalog, index, axis=0)

del(do_not_exist)
gc.collect()

total_nb_images = np.shape(gz2_catalog)[0]

class_list = ["A", "Ec", "Ei", "Er", "SBa", "SBb", "SBc", "SBd", "Sa", "Sb", "Sc", "Sd", "Se"]
class_list = ["E", "SB", "S"]

frac_test = 0.05

nb_class = int(len(class_list))
class_count_train = np.zeros(nb_class, dtype="int")
class_count_test = np.zeros(nb_class, dtype="int")

for i in range(0, nb_class):
	index = np.where(gz2_catalog[:,2].astype("<U%d"%(len(class_list[i]))) == class_list[i])[0]
	class_count_test[i] = int(frac_test*np.shape(index)[0])
	class_count_train[i] = np.shape(index)[0] - class_count_test[i]

total_nb_train = np.sum(class_count_train)
total_nb_test = np.sum(class_count_test)

filename_to_class_array_train = np.zeros((total_nb_train,3), dtype="int")
filename_to_class_array_test = np.zeros((total_nb_test,3), dtype="int")

train_cumsum = np.zeros(nb_class+1, dtype="int")
test_cumsum = np.zeros(nb_class+1, dtype="int")

train_cumsum[1:] = np.cumsum(class_count_train)
test_cumsum[1:] = np.cumsum(class_count_test)

for i in range(0, nb_class):
	index = np.where(gz2_catalog[:,2].astype("<U%d"%(len(class_list[i]))) == class_list[i])[0]

	filename_to_class_array_train[train_cumsum[i]:train_cumsum[i+1], 0] = gz2_catalog[index[:class_count_train[i]],1]
	filename_to_class_array_train[train_cumsum[i]:train_cumsum[i+1], 1] = i
	filename_to_class_array_test[test_cumsum[i]:test_cumsum[i+1], 0] = gz2_catalog[index[class_count_train[i]:],1]
	filename_to_class_array_test[test_cumsum[i]:test_cumsum[i+1], 1] = i

crop_size = int(np.maximum(raw_image_size//2,image_size))

transform = A.Compose([
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
])

training_rebalance = np.clip(class_count_train[:],0,4000).astype("float")
print(training_rebalance)
training_rebalance /= np.sum(training_rebalance)

nb_im_train = 2048
nb_im_test = total_nb_test

input_data = np.zeros((nb_im_train,image_size*image_size*(im_depth+1)), dtype="float32")
targets = np.zeros((nb_im_train,nb_class), dtype="float32")

input_test = np.zeros((nb_im_test,image_size*image_size*(im_depth+1)), dtype="float32")
targets_test = np.zeros((nb_im_test,nb_class), dtype="float32")

zero_target = np.zeros(nb_class)

def create_train_batch(visual=0):

	if(visual):
		fig, axs = plt.subplots(4,5, figsize=(5,4), dpi=250, constrained_layout=True)

	for i in range(0,nb_im_train):
		r_class = np.random.choice(np.arange(0,nb_class), p=training_rebalance)
		
		i_d = train_cumsum[r_class] + int(np.random.random()*class_count_train[r_class])
		path = "gz2_images_%dx%d/%d.jpg"%(image_size,image_size,filename_to_class_array_train[i_d,0])

		patch = np.asarray(Image.open(path))
		transformed = transform(image=patch)
		patch = (transformed['image']/255.0)

		for depth in range(0,im_depth):
			input_data[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = (np.copy(patch[:,:,depth]).flatten("C"))
		input_data[i,3*image_size*image_size:] = 0.0
		targets[i,:] = 0
		targets[i,filename_to_class_array_train[i_d,1]] = 1

		if(visual and i < 20):
			axs[i//5][i%5].imshow(patch)
			axs[i//5][i%5].set_axis_off()
			axs[i//5][i%5].text(0.1,0.1, "%s"%(class_list[filename_to_class_array_train[i_d,1]]), c="limegreen", va="top", fontsize=6)

	if(visual):
		plt.savefig("training_set_example.jpg", dpi=250)
		return

	return input_data, targets


def create_test_batch():

	for i in range(0,nb_im_test):
		path = "gz2_images_%dx%d/%d.jpg"%(image_size, image_size, filename_to_class_array_test[i,0])

		patch = np.asarray(Image.open(path))/255.0

		for depth in range(0,im_depth):
			input_test[i,depth*image_size*image_size:(depth+1)*image_size*image_size] = (np.copy(patch[:,:,depth]).flatten("C"))
		input_test[i,3*image_size*image_size:] = 0.0
		targets_test[i,:] = 0
		targets_test[i,filename_to_class_array_test[i,1]] = 1

	return input_test, targets_test

