# from keras.models import load_model
# model = load_model('saved_model_10epochs.h5')
from matplotlib import pyplot
from numpy import vstack, expand_dims
from tensorflow.keras.preprocessing.image import  img_to_array

def preprocess_data(X):
	# load compressed arrays
	# unpack arrays

	# scale from [0,255] to [-1,1]
	X = img_to_array(X)
	X = (X - 127.5) / 127.5
	print(f"Image shape after preprocessing: {X.shape}") 
	image_array = expand_dims(X, axis=0)
    
    # Verify the shape before prediction
    # print(f"Image shape before prediction: {image_array.shape}") 


	

	return image_array
	


	

def plot_images(src_img, gen_img):
	images = vstack((src_img, gen_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()





# plot all three images
