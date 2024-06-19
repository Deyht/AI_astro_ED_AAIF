
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os, glob
from threading import Thread
import time
import albumentations as A
from PIL import Image
import cv2
import pickle


processed_data_path = "./"

nb_images_per_iter = 4096
nb_class = 10

nb_train = nb_class*5000
nb_val = nb_class*1000


nb_workers = 2 #Need to be a factor of nb_images_per_iter

image_size_raw = 32

image_size = 32
image_size_val = image_size
max_scale = image_size
min_scale = image_size

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def init_data_gen(test_mode = 0):
	
	global nb_images_per_iter, nb_val, image_size, image_size_val, max_scale, min_scale
	global flat_image_slice, nb_workers, block_size, raw_train_data, raw_train_targets, raw_val_data, raw_val_targets
	global input_data, targets, input_val, targets_val, targets_zero, nb_process, nb_class, class_list

	np.random.seed(int(time.time()))

	flat_image_slice = image_size*image_size

	block_size = nb_images_per_iter // nb_workers
	
	if(not os.path.isdir("cifar-10-batches-py")):
		os.system("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
		os.system("tar -xzf cifar-10-python.tar.gz")
	
	targets_zero = np.zeros((nb_class), dtype="float32")
	
	batches_meta = unpickle("cifar-10-batches-py/batches.meta")
	class_list = batches_meta[b"label_names"]
	for i in range(0,nb_class):
		class_list[i] = class_list[i][:5]
	
	if(test_mode == 0): #used to control the behavior regarging training, validation and test 
	
		raw_train_data = np.zeros((nb_train, flat_image_slice*3), dtype="uint8")
		raw_train_targets = np.zeros((nb_train), dtype="uint8")
	
		i = 0
		for batch in glob.glob("cifar-10-batches-py/data_batch_*"):
			dict_batch = unpickle(batch)
			raw_train_data[i*10000:(i+1)*10000,:] = dict_batch[b"data"]
			raw_train_targets[i*10000:(i+1)*10000] = dict_batch[b"labels"]
			i += 1
	
		input_data = np.zeros((nb_images_per_iter,flat_image_slice*3), dtype="float32")
		targets = np.zeros((nb_images_per_iter,nb_class), dtype="float32")

	
	raw_val_data = np.zeros((nb_val, flat_image_slice*3), dtype="uint8")
	raw_val_targets = np.zeros((nb_val), dtype="uint8")
	
	dict_batch = unpickle("cifar-10-batches-py/test_batch")
	raw_val_data[:,:] = dict_batch[b"data"]
	raw_val_targets[:] = dict_batch[b"labels"]
	
	input_val = np.zeros((nb_val,flat_image_slice*3), dtype="float32")
	targets_val = np.zeros((nb_val,nb_class), dtype="float32")


def create_train_aug(i, rf_id, rf_scale):

	patch = np.zeros((image_size, image_size,3), dtype="uint8")

	for l in range(0,block_size):

		r_id = int(rf_id[l]*nb_train)
		
		for depth in range(0,3):
			patch[:,:,depth] = np.reshape(raw_train_data[r_id,depth*flat_image_slice:(depth+1)*flat_image_slice], (32,32))
		
		l_scale = int(rf_scale[l]*(max_scale-min_scale)+min_scale)
		l_translate = int(np.maximum(0,(image_size+(image_size/4.0)-l_scale)*0.5))
		
		transform = A.Compose([
			#Affine here act more as an aspect ratio transform than scaling
			#A.Affine(scale=(0.9,1.1), translate_px=(-4,4), rotate=(-10,10), fit_output=False, interpolation=1, p=1.0),
			
			#Un-comment for scale augment 
			#A.SmallestMaxSize(max_size=l_scale, interpolation=1, p=1.0),
			#A.Affine(translate_px=(-l_translate,l_translate),p=1.0),
			#A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
			#A.RandomCrop(width=image_size, height=image_size, p=1.0),

			A.HorizontalFlip(p=0.5),
			
			#A.ColorJitter(brightness=(0.75,1.33), contrast=(0.75,1.33), saturation=(0.75,1.33), hue=0.1, p=1.0),
			#A.ToGray(p=0.02),
			
			#A.OneOf([
	        #	A.ISONoise(p=0.1),
	        #	A.MultiplicativeNoise(per_channel=False, elementwise=True, p=2.0),
	        #	A.GaussNoise(var_limit=(0.0,0.03*255), per_channel=False, p=0.5),
	        #	A.PixelDropout(dropout_prob=0.03, per_channel=False, p=2.0),
	        #	A.ImageCompression(quality_lower=20, quality_upper=40, p=1.0),
			#	A.GaussianBlur(p=1.0),
			#], p=0.05),
		#Various types of noise / image alterations. Tend to reduce the validation accuracy, but build a more resilient network for deployement or other applications
		])
		
		transformed = transform(image=patch)

		patch_aug = transformed['image']
		
		for depth in range(0,3):
			input_data[i[l],depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:, depth].flatten("C") - 100.0)/155.0
		
		targets[i[l],:] = np.copy(targets_zero[:])
		targets[i[l],raw_train_targets[r_id]] = 1.0
	
	
def visual_aug(visual_w, visual_h):
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_data[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch, interpolation="lanczos")
		ax[c_x,c_y].axis('off')
		
		p_c = np.argmax(targets[i,:])
		
		c_text = ax[c_x,c_y].text(15/256*image_size, 25/256*image_size, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=11, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
	
	plt.savefig("augm_mosaic.jpg", dpi=200)


def create_train_batch():
	
	nb_blocks = nb_images_per_iter / block_size

	i_d = np.arange(nb_images_per_iter)
	rf_id = np.random.random(nb_images_per_iter)
	rf_scale = np.random.random(nb_images_per_iter)
	
	t_list = []
	b_count = 0
	
	for k in range(0,nb_workers):
		t = Thread(target=create_train_aug, args=[i_d[b_count*block_size:(b_count+1)*block_size], \
												rf_id[b_count*block_size:(b_count+1)*block_size], \
												rf_scale[b_count*block_size:(b_count+1)*block_size]])
		t.start()
		t_list = np.append(t_list, t)
		b_count += 1
	
	for k in range(0,nb_workers):
		t_list[k].join()
	
	return input_data, targets


def create_val_batch():
	print("Loading validation data ...")

	for i in range(nb_val):
		
		input_val[i,:] = (raw_val_data[i] - 100.0)/155.0
		
		targets_val[i,:] = np.copy(targets_zero[:])
		targets_val[i,raw_val_targets[i]] = 1.0
	
	return input_val, targets_val


def visual_val(visual_w, visual_h):
	
	#Loading the full validation dataset can take a while => nb_val can be significantly reduced for faster visualization
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch, interpolation="lanczos")
		ax[c_x,c_y].axis('off')
		
		p_c = np.argmax(targets_val[i,:])
		
		c_text = ax[c_x,c_y].text(15/256*image_size, 25/256*image_size, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=11, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
	
	plt.savefig("val_mosaic.jpg", dpi=200)

	
def visual_pred(load_epoch=0, visual_w=8, visual_h=6):

	pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%(load_epoch), dtype="float32")
	predict = np.reshape(pred_raw, (nb_val, -1))
	patch = np.zeros((image_size,image_size,3), dtype="uint8")

	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	
	for i in range(0, visual_w*visual_h):
		
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			patch[:,:,depth] = np.reshape(raw_val_data[i,depth*flat_image_slice:(depth+1)*flat_image_slice], (image_size_val,image_size_val))
		
		ax[c_x,c_y].imshow(patch, interpolation="lanczos")
		ax[c_x,c_y].axis('off')
		
		p_c = raw_val_targets[i]
		
		c_text = ax[c_x,c_y].text(15/256*image_size, 25/256*image_size, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=11, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
		
		p_c = np.argmax(predict[i])
		
		c_text = ax[c_x,c_y].text(15/256*image_size_val, image_size_val - 20/256*image_size_val, 
			"Pred: %s - %0.3f"%(class_list[p_c], predict[i,p_c]), c=plt.cm.tab20(p_c%20), fontsize=11, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
		
	plt.savefig("pred_mosaic.jpg", dpi=200)


