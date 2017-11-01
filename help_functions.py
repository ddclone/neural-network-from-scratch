import numpy as np

def make_spirals(n_rows, spirals, theta):
    N, K = n_rows, spirals
    x = np.zeros((N*K,2), dtype='float32')
    y = np.zeros((N*K,1), dtype='uint8')
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*5,(j+1)*5,N) + np.random.randn(N)*theta # theta
        x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return x, y

def make_flower(n_rows, rays, theta, radius):
    m, a = n_rows, rays
    N = int(m/2) 
    x = np.zeros((m,2), dtype='float32') 
    y = np.zeros((m,1), dtype='uint8') 
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*theta 
        r = a*np.sin(4*t) + np.random.randn(N)*radius
        x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return x, y

def one_hot_encoding(target):
    T = np.zeros((len(target), len(np.unique(target))), dtype='float32')
    for i in range(len(target)):
        T[i, target[i,]] = 1.0
    return T
            
def normalization(x):
    train_mean = np.mean(x, axis = 0, keepdims=True)
    x_cen = x - train_mean
    train_std = np.var(x_cen, axis = 0, keepdims=True)
    x_std = x_cen/(train_std+1e-8)
    return x_std, train_mean, train_std

def create_batches(x, y, mb_size):
    indexes = np.arange(len(x)) 
    np.random.shuffle(indexes)
    a, b = x[indexes], y[indexes]
    minibatch = [ [a[k:k + mb_size], b[k:k + mb_size]] for k in range(0, len(x), mb_size)]
    return minibatch

def make_grid(x, grid_size, border):
    x_min, x_max = x[:, 0].min() - border, x[:, 0].max() + border
    y_min, y_max = x[:, 1].min() - border, x[:, 1].max() + border
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_size),
                         np.arange(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return (xx.astype('float32'), yy.astype('float32'), grid.astype('float32'))
