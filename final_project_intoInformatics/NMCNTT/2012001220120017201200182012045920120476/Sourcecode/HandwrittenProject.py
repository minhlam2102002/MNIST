import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
from sklearn import neighbors
from numba import jit

def load_mnist(path, kind):
	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
	images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

	with gzip.open(labels_path, 'rb') as lbpath:
		lbpath.read(8)
		buffer = lbpath.read()
		labels = np.frombuffer(buffer, dtype = np.uint8)
	with gzip.open(images_path, 'rb') as imgpath:
		imgpath.read(16)
		buffer = imgpath.read()
		images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28, 28).astype(np.float64)
	
	return images, labels

X_train, y_train = load_mnist('data/','train')
X_test, y_test = load_mnist('data/','t10k')
print('Rows: %d, colums: %d' % (X_train.shape[0], X_train.shape[1]))

fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True,)
ax = ax.flatten()
for i in range (10):
	img = X_train[y_train == i][0]
	ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Thong so
start = 0                             
end = X_train.shape[0]        
number_of_neighbors = 99           # k for kNN
sampling_type = 'avg'              # sampling type      
sampling_subsize = 2               # sampling subsize  
histogram_max = 255                # histogram maximum value

# Rut trich dac trung bang cach vector hoa anh

def vectorization(X_train):
    size, row, col = X_train.shape
    return X_train.reshape(size, row * col)
    

X_train_vector = vectorization(X_train)
print("\nVector hoa:", X_train_vector.shape)

# Rut trich dac trung bang cach sampling

@jit
def sampling_max(X_train, subsize):
    size, row, col = X_train.shape
    convert = np.zeros((size, int(row / subsize), int(col / subsize)), dtype = np.uint8)
    for i in range(size):
        for x in range(int(row / subsize) - 1):
            for y in range(int(col / subsize) - 1):
                convert[i][x][y] = np.amax(X_train[i][x * subsize : (x + 1) * subsize, y * subsize : (y + 1) * subsize]) 
    return convert

@jit
def sampling_min(X_train, subsize):
    size, row, col = X_train.shape
    convert = np.zeros((size, int(row / subsize), int(col / subsize)), dtype = np.uint8)
    for i in range(size):
        for x in range(int(row / subsize) - 1):
            for y in range(int(col / subsize) - 1):
                convert[i][x][y] = np.amin(X_train[i][x * subsize : (x + 1) * subsize, y * subsize : (y + 1) * subsize]) 
    return convert

@jit
def sampling_avg(X_train, subsize):
    size, row, col = X_train.shape
    convert = np.zeros((size, int(row / subsize), int(col / subsize)), dtype = np.float64)
    for i in range(size):
        for x in range(int(row / subsize) - 1):
            for y in range(int(col / subsize) - 1):
                convert[i][x][y] = np.sum(X_train[i][x * subsize : (x + 1) * subsize, y * subsize : (y + 1) * subsize]) / (subsize * subsize);
    return convert

def sampling(X_train, subsize, function):
    if (function == 'max'): 
        return sampling_max(X_train, subsize)
    elif (function == 'min'): 
        return sampling_min(X_train, subsize)
    elif (function == 'avg'): 
        return sampling_avg(X_train, subsize)

X_train_sampling = sampling(X_train, sampling_subsize, sampling_type)
print("\nSampling:", X_train_sampling.shape)

# Rut trich dac trung bang cach histogram

@jit
def histogram(X_train):
    size, row, col = X_train.shape
    convert = np.zeros((size, histogram_max), dtype = np.uint8)
    for i in range(size):
        convert[i], temp = np.histogram(X_train[i], bins = np.arange(histogram_max + 1))   
    return convert
    

X_train_histogram = histogram(X_train)
print("\nHistogram:", X_train_histogram.shape)
    
# Su dung thu vien sklearn

def runningknn(X_train_vector, y_train, X_test_vector, y_test, numofnei):
    clf = neighbors.KNeighborsClassifier(n_neighbors = numofnei)
    clf.fit(X_train_vector[start:end], y_train[start:end])
    
    scikit_pred = clf.predict(X_test_vector[start:end])
    print('kNN method correct rate =', np.sum(scikit_pred == y_test[start:end]) / 1.0 / scikit_pred.shape[0])
    
def runningmean(X_train_vector, y_train, X_test_vector, y_test):
    clf = neighbors.NearestCentroid()
    clf.fit(X_train_vector[start:end], y_train[start:end])
    
    scikit_pred = clf.predict(X_test_vector[start:end])
    print('Mean method correct rate =', np.sum(scikit_pred == y_test[start:end]) / 1.0 / scikit_pred.shape[0])

#Vector hoa
print('\nVector hoa:')
X_test_vector = vectorization(X_test)
runningknn(X_train_vector, y_train, X_test_vector, y_test, number_of_neighbors)
runningmean(X_train_vector, y_train, X_test_vector, y_test)

#Sampling
print('\nSampling:')
X_test_sampling = sampling(X_test, sampling_subsize, sampling_type)
X_train_sampling_reshape = vectorization(X_train_sampling)
X_test_sampling_reshape = vectorization(X_test_sampling)
runningknn(X_train_sampling_reshape, y_train, X_test_sampling_reshape, y_test, number_of_neighbors)
runningmean(X_train_sampling_reshape, y_train, X_test_sampling_reshape, y_test)

#Histogram
print('\nHistogram:')
X_test_histogram = histogram(X_test)
runningknn(X_train_histogram, y_train, X_test_histogram, y_test, number_of_neighbors)
runningmean(X_train_histogram, y_train, X_test_histogram, y_test)