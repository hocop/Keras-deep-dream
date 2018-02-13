import numpy as np
import keras
import glob
import cv2
import time

from keras import Input
from keras import backend as K

model_size = 224
classes = 1000
amp_index = 0
iterations = 50
rate = 1e-2
depth = 1
rgb = True

_grad = None
_func = None

K.set_learning_phase(1) #set learning phase (костыль какой-то)

def swapChannels(x):
	b,g,r = cv2.split(x)
	return cv2.merge((r,g,b))

# create deep dream function
def getGradFunc(model):
	_amplify = Input(shape=(classes,))
	_inputs = Input(shape=(model_size,model_size,3))
	_x = model(_inputs)
	_errgrad = _x - _amplify
	_errors = tf.reduce_sum(tf.square(_errgrad))
	_grads = K.gradients(_errors, _inputs)
	_grad = K.function([_inputs,_amplify], _grads)
	return _grad

def dream(img, feature, _grad, iterations=10, rate=1e-2, show=True):
	def get_grads(img):
		arg1 = img.reshape(1,model_size,model_size,3)
		arg2 = np.array(feature).reshape(1,len(feature))
		grad = _grad([arg1,arg2])
		grad = np.array(grad).reshape(model_size,model_size,3)
		return grad
	img0 = np.array(img)
	width, height, c = img.shape
	for i in range(iterations):
		grads = np.zeros(img.shape)
		for d in range(1,depth+1):
			w = width // d
			h = height // d
			for iw in range(d):
				for ih in range(d):
					smallimg = img[iw*w:(iw+1)*w,ih*h:(ih+1)*h]
					smallimg = cv2.resize(smallimg,(model_size,model_size))
					smallgrad = get_grads(smallimg)
					smallgrad = cv2.resize(smallgrad,(h,w))
					smallgrad /= np.mean(np.abs(smallgrad)) + 1e-6
					smallgrad /= d**2
					grads[iw*w:(iw+1)*w,ih*h:(ih+1)*h] += smallgrad
		scale = np.mean(np.abs(grads))+1e-6
		grads /= scale
		img -= grads * rate
		
		if show:
			frame0 = np.array(img0)
			frame = np.array(img)
			if rgb:
				frame = swapChannels(frame)
				frame0 = swapChannels(frame0)
			cv2.imshow('frame0',frame0)
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	return img

if __name__ == '__main__':
	path = './'
	imnames = glob.iglob(path + 'examples/*', recursive=True)
	imnames = sorted(imnames)
	
	model = keras.applications.vgg16.VGG16(
			input_shape=(model_size,model_size,3),
			weights='imagenet',
			include_top=True)
	
	feature = [0. for i in range(classes)]
	feature[amp_index] = 1.
	
	_grad = getGradFunc(model)
	for name in imnames:
		print(name)
		img = cv2.imread(name) / 255.0
		if rgb:
			img = swapChannels(img)
		img = dream(img, feature, _grad, iterations, rate, False)
		if rgb:
			img = swapChannels(img)
		cv2.imwrite('outputs/'+name.split('/')[-1], img*255)




