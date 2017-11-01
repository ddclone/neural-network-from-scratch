import numpy as np
import help_functions as hf
from network_class import neural_network
import matplotlib.pyplot as plt

spirals = 4
X,Y = hf.make_spirals(n_rows=100, spirals=spirals, theta=0.11)

x, train_mean, train_var = hf.normalization(X)
y = hf.one_hot_encoding(Y)
N = len(x)
minibatch_size = 64

model = neural_network(size=[x.shape[1], 128, 128,  y.shape[1]],
                       activations=['linear', 'relu', 'relu', 'softmax'],
                       dropout=[1.0, 0.75, 0.75, 1.0], 
                       lambda_reg=1e-4)
model.init_params(N)
model.summary()
model.train_mean, model.train_var = train_mean, train_var

init_rate = 0.001
rate = float(init_rate)

losses = []
smooth_losses = []
smooth_loss = 0.0
epochs = 1000

for epoch in range(1, epochs+1):
    
    if rate >= 0.0001: rate = init_rate*0.999**epoch 
    loss = 0.0

    minibatches = hf.create_batches(x, y, minibatch_size)
    for mb in minibatches:
        loss += model.run_epoch(mb[0], mb[1], rate) 
    
    if epoch != 1: smooth_loss = 0.95*smooth_loss + 0.05*loss
    else: smooth_loss = loss
    
    losses.append(loss)
    smooth_losses.append(smooth_loss)
    if epoch % 100 == 0:
        print('Epoch: %i/%i, Rate: %.4f, Loss: %.4f'
              % (epoch, epochs, rate, loss,))


xx, yy, grid = hf.make_grid(X, grid_size=0.01, border=0.15)
prediction = model.predict(grid)
prediction = np.argmax(prediction,axis=1).reshape(xx.shape) 

fig, ax = plt.subplots( nrows=1, ncols=1 ) 
ax.contourf(xx, yy, prediction, alpha=0.75, cmap=plt.cm.jet)
ax.scatter(X[:,0],  X[:,1], c = Y[:,0], s=40, cmap=plt.cm.jet, edgecolor='black')
fig.savefig('test.png')   
plt.close(fig)

fig, ax = plt.subplots( nrows=1, ncols=1 ) 
plt.subplot(2,1,1)
line1, = plt.plot(np.arange(epochs), losses, c='blue', label='loss')
plt.legend(handles=[line1], loc=1)
plt.subplot(2,1,2)
line2, = plt.plot(np.arange(epochs), smooth_losses, c='red', label='smooth loss')
plt.legend(handles=[line2], loc=1)
fig.savefig('losses.png')   
plt.close(fig)
