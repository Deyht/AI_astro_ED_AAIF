
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os, glob
from threading import Thread
import time
import albumentations as A
from PIL import Image
import cv2


processed_data_path = "./"

nb_images_per_iter = 4096
nb_class = 2

nb_keep_val = nb_class*1024


nb_workers = 2 #Need to be a factor of nb_images_per_iter

image_size_raw = 128

image_size = 128
image_size_val = image_size
max_scale = image_size+32
min_scale = image_size-32

class_count = 12500
class_list = ["Cat", "Dog"]

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")
	

def init_data_gen(test_mode = 0):
	
	global nb_images_per_iter, nb_keep_val, image_size, image_size_val, max_scale, min_scale
	global flat_image_slice, nb_workers, block_size, transform_val, raw_data_array
	global input_data, targets, input_val, targets_val, targets_zero, nb_process, nb_class, class_list

	np.random.seed(int(time.time()))

	flat_image_slice = image_size*image_size

	block_size = nb_images_per_iter // nb_workers

	transform_val = A.Compose([
		A.LongestMaxSize(max_size=image_size_val, interpolation=1, p=1.0),
		A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
	])
	
	if(not os.path.isfile("asirra_bin_128.dat")):
		os.system("wget https://share.obspm.fr/s/6TBsCpAASeETH3S/download/asirra_bin_128.tar.gz")
		os.system("tar -xzf asirra_bin_128.tar.gz")
	
	targets_zero = np.zeros((nb_class), dtype="float32")
	
	raw_data_array = np.reshape(np.fromfile("asirra_bin_128.dat", dtype="uint8"), (class_count*2,image_size_raw,image_size_raw,3))
	
	if(test_mode == 0): #used to control the behavior regarging training, validation and test 
		input_data = np.zeros((nb_images_per_iter,flat_image_slice*3), dtype="float32")
		targets = np.zeros((nb_images_per_iter,nb_class), dtype="float32")
	
	input_val = np.zeros((nb_keep_val,flat_image_slice*3), dtype="float32")
	targets_val = np.zeros((nb_keep_val,nb_class), dtype="float32")



def create_train_aug(i, rf_c, rf_id, rf_scale):

	for l in range(0,block_size):

		r_class = int(rf_c[l]*nb_class)
		r_id = int(rf_id[l]*(class_count-1024))
		
		patch = raw_data_array[r_class*class_count+r_id]
		
		l_scale = int(rf_scale[l]*(max_scale-min_scale)+min_scale)
		l_translate = int(np.maximum(0,(image_size+(image_size/14.0)-l_scale)*0.5))
		
		transform = A.Compose([
			#Affine here act more as an aspect ratio transform than scaling
			A.Affine(scale=(0.85,1.15), rotate=(-10,10), fit_output=True, interpolation=1, p=1.0),

			#No scale augment
			#A.LongestMaxSize(max_size=image_size_val, interpolation=1, p=1.0),
			#A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),

			#Un-comment for scale augment 
			A.SmallestMaxSize(max_size=l_scale, interpolation=1, p=1.0),
			A.Affine(translate_px=(-l_translate,l_translate),p=1.0),
			A.PadIfNeeded(min_width=image_size, min_height=image_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
			A.RandomCrop(width=image_size, height=image_size, p=1.0),

			A.HorizontalFlip(p=0.5),
			
			A.ColorJitter(brightness=(0.75,1.3), contrast=(0.75,1.3), saturation=(0.75,1.3), hue=0.15, p=1.0),
			A.ToGray(p=0.02),
			
			#A.OneOf([
	        #	A.ISONoise(p=0.1),
	        #	A.MultiplicativeNoise(per_channel=False, elementwise=True, p=2.0),
	        #	A.GaussNoise(var_limit=(0.0,0.03*255), per_channel=False, p=0.5),
	        #	A.PixelDropout(dropout_prob=0.03, per_channel=False, p=2.0),
	        #	A.ImageCompression(quality_lower=20, quality_upper=40, p=1.0),
			#	A.GaussianBlur(p=1.0),
			#], p=0.0),
		#Various types of noise / image alterations. Tend to reduce the validation accuracy, but build a more resilient network for deployement or other applications
		])
		
		transformed = transform(image=patch)

		patch_aug = transformed['image']
		
		for depth in range(0,3):
			input_data[i[l],depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0
		
		targets[i[l],:] = np.copy(targets_zero[:])
		targets[i[l],r_class] = 1.0
	
	
def visual_aug(visual_w, visual_h):
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_data[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis('off')
		
		p_c = np.argmax(targets[i,:])
		
		c_text = ax[c_x,c_y].text(15/256*image_size, 25/256*image_size, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=12, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
	
	plt.savefig("augm_mosaic.jpg", dpi=200)


def create_train_batch():
	
	nb_blocks = nb_images_per_iter / block_size

	i_d = np.arange(nb_images_per_iter)
	rf_c = np.random.random(nb_images_per_iter)
	rf_id = np.random.random(nb_images_per_iter)
	rf_scale = np.random.random(nb_images_per_iter)
	
	t_list = []
	b_count = 0
	
	for k in range(0,nb_workers):
		t = Thread(target=create_train_aug, args=[i_d[b_count*block_size:(b_count+1)*block_size], \
												rf_c[b_count*block_size:(b_count+1)*block_size], \
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

	for i in range(nb_keep_val):
		
		if((i+1)%2 != 0):
			patch = raw_data_array[class_count - 1024 + i//2]
			r_class = 0
		else:
			patch = raw_data_array[class_count*2 - 1024 + i//2]
			r_class = 1
		
		transformed = transform_val(image=patch)
		patch_aug = transformed['image']
		
		for depth in range(0,3):
			input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice] = (patch_aug[:,:,depth].flatten("C") - 100.0)/155.0
		
		targets_val[i,:] = np.copy(targets_zero[:])
		targets_val[i,r_class] = 1.0
	
	return input_val, targets_val


def visual_val(visual_w, visual_h):
	
	#Loading the full validation dataset can take a while => nb_keep_val can be significantly reduced for faster visualization
	
	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)
	l_patch = np.zeros((image_size, image_size, 3))
	
	for i in range(0, visual_w*visual_h):
		c_x = i // visual_w
		c_y = i % visual_w
		
		for depth in range(0,3):
			l_patch[:,:,depth] = np.reshape((input_val[i,depth*flat_image_slice:(depth+1)*flat_image_slice]*155.0 + 100.0)/255.0, (image_size, image_size))
		ax[c_x,c_y].imshow(l_patch)
		ax[c_x,c_y].axis('off')
		
		p_c = int((i)%2)
		
		c_text = ax[c_x,c_y].text(15/256*image_size, 25/256*image_size, "%s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=12, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
	
	plt.savefig("val_mosaic.jpg", dpi=200)

	
def visual_pred(load_epoch=0, visual_w=8, visual_h=6):

	pred_raw = np.fromfile("fwd_res/net0_%04d.dat"%(load_epoch), dtype="float32")
	predict = np.reshape(pred_raw, (nb_keep_val,nb_class))

	fig, ax = plt.subplots(visual_h, visual_w, figsize=(2*visual_w,2*visual_h), dpi=200, constrained_layout=True)

	for i in range(0, visual_w*visual_h):
		
		if((i+1)%2 != 0):
			patch = raw_data_array[class_count - 1024 + i//2]
			r_class = 0
		else:
			patch = raw_data_array[class_count*2 - 1024 + i//2]
			r_class = 1

		transformed = transform_val(image=patch)
		patch_aug = transformed['image']
		
		c_x = i // visual_w
		c_y = i % visual_w
		
		ax[c_x,c_y].imshow(patch_aug)
		ax[c_x,c_y].axis('off')
		
		p_c = int(r_class)
		
		c_text = ax[c_x,c_y].text(15/256*image_size, image_size - 20/256*image_size, "Targ: %s"%(class_list[p_c]), c=plt.cm.tab20(p_c%20), fontsize=12, clip_on=True)
		c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
		
		for k in range(0, nb_class):
			c_text = ax[c_x,c_y].text(10/256*image_size, (20+k*25)/256*image, "%0.2f - %s"%(pred_values[k], class_list[ind_sort[k]]), 
				c=plt.cm.tab20(ind_sort[k]%20), fontsize=6, clip_on=True)
			c_text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()])
			
	plt.savefig("pred_mosaic.jpg", dpi=200)


def free_data_gen():
  global input_data, targets, input_val, targets_val
  del (input_data, targets, input_val, targets_val)
  return



