import os
import cv2
import glob
import tqdm
import logging
import argparse
from multiprocessing import Pool

# special_char = ['?', '-', '.', '*', '@', '!', '#', '$', '%', '^' '&', '(', ')']
__author__ = 'cristian'

def check_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)

def processing(dataset, new_h=10):
	"""

	"""
	logging.basicConfig(filename='logs/logs_processing.txt',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
	# logging.info("Logging processing data")
	#
	all_datasets = os.listdir(dataset)
	print(all_datasets)
	for data in all_datasets:
		if data == '540k':
			images_path = dataset + '/' + data + '/' + 'images'
			for image in tqdm.tqdm(os.listdir(images_path)):
				try:
					path_image = os.path.join(images_path, image)
					name = image.split('_')[-1]
					img = cv2.imread(path_image)
					h, w = img.shape[0], img.shape[1]
					# print(img.shape)
					new_w = int((new_h*w)/h)
					new_shape = (new_w, new_h)
					# print(new_shape)
					new_img = cv2.resize(img, new_shape, interpolation = cv2.INTER_AREA)
					saver_path = dataset + '/' + data + '/' + 'processed'
					check_exists(saver_path)
					result = cv2.imwrite(saver_path + '/' + name, new_img)
				except Exception as e:
					logging.error("Exception preprocessing image ", exc_info=True)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Pre-processing dataset')

	parser.add_argument(
		"-data",
		"--dataset",
		type=str,
		nargs="?",
		help="The input directory",
		default="./datasets",
	)
	args = parser.parse_args()
	check_exists('logs')
	processing(args.dataset)


