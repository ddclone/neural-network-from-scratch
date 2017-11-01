import numpy as np

class neural_network:
    
    np.random.seed(10)
    
    def __init__(self, size, activations, dropout=[], lambda_reg=0.0):
        self.size = size
        self.activations = activations
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.depth = len(self.size)
        self.epsilon = 1e-8
        
        self.Vd_weights = [0.0 for _ in self.size[1:]]
        self.Sd_weights = [0.0 for _ in self.size[1:]]
        self.Vd_biases = [0.0 for _ in self.size[1:]]
        self.Sd_biases =  [0.0 for _ in self.size[1:]]
    
    def init_params(self, n):
        self.weights = [(np.random.randn(a, b)*2.0/np.sqrt(n)).astype('float32') 
                        for a, b in zip(self.size[:-1], self.size[1:])]
        self.biases = [np.zeros((1, b), dtype='float32')  for b in self.size[1:]]
    
    def summary(self):
         p = sum([w.shape[0]*w.shape[1] for w in self.weights]) 
         d = sum([b.shape[0]*b.shape[1] for b in self.biases])
         print('Trainable parameters: ' + str(p+d))
         
    def activation(self, x, name):
        if name=='relu': return np.maximum(0, x)
        elif name=='softmax': 
            e = np.exp(x) 
            return e/np.sum(e, axis=1, keepdims=True)
        elif name=='linear': return x
    
    def derivative_activaion(self, dl, l, name, n=None):
        if name=='relu': 
            dl[l <= 0.0] = 0.0
            return dl
        elif name == 'linear' or name == 'softmax': return dl  
        
    def forward_propagation(self, x):
        self.layers = [x]
        self.layers.extend([[] for _ in self.size[1:]])
        self.dropout_mask = []
        for i in range(1, self.depth):
            z = np.dot(self.layers[i-1], self.weights[i-1]) + self.biases[i-1]        
            self.layers[i] = self.activation(z, self.activations[i])
            if len(self.dropout) > 1:
                if self.dropout[i] < 1.0:
                   keep_layer = np.random.rand(self.layers[i].shape[1]) > self.dropout[i]
                   self.layers[i][:,keep_layer] = 0.0
                   self.layers[i] /= self.dropout[i]
                   self.dropout_mask.append(keep_layer)
                else:
                    self.dropout_mask.append([])
                     
    def compute_loss(self, y):
        if self.activations[self.depth-1] == 'softmax':
            dataloss = np.sum(-np.log(self.layers[self.depth-1]+self.epsilon)*y)/len(y)
        elif self.activations[self.depth-1] == 'linear':
            dataloss = np.sum((self.layers[self.depth-1]-y)**2)/len(y)
        regloss =  self.lambda_reg/(2*len(y)) * sum([ np.sum(w)**2  for w in self.weights]) 
        return dataloss + regloss
    
    def backward_propagation(self, y):
        m = np.float32(1.0/len(y))   
        lambda_reg = self.lambda_reg * m
        self.d_layers = [[] for _ in self.size[1:]]
        self.d_layers.extend([ (self.layers[len(self.layers)-1] - y) ])
        
        self.d_weights = [[] for _ in self.size[1:]]
        self.d_biases = [[] for _ in self.size[1:]]
        for i in reversed(range(1, self.depth)):
            l_reg = lambda_reg*self.weights[i-1]
            self.d_weights[i-1] = m*np.dot(self.layers[i-1].T, self.d_layers[i]) + l_reg 
            self.d_biases[i-1] = m*np.sum(self.d_layers[i], axis=0, keepdims=True)
            if i != 1:
                self.d_layers[i-1] = np.dot(self.d_layers[i], self.weights[i-1].T)   
                self.d_layers[i-1] = self.derivative_activaion(self.d_layers[i-1], self.layers[i-1],
                                                               self.activations[i-1]) 
                if len(self.dropout) > 1:
                    if self.dropout[i-1] != 1.0:
                        self.layers[i][:,self.dropout_mask[i-1]] = 0.0
                        self.layers[i] /= self.dropout[i]
     
#    def gradient_descent(self, rate):
#        for i in range(self.depth-1):
#            self.weights[i] = self.weights[i] - rate * self.d_weights[i]
#            self.biases[i] = self.biases[i] - rate * self.d_biases[i]
            
    def adam_optimizer(self, rate, beta1, beta2):
        for i in range(self.depth-1):
            self.Vd_weights[i] = beta1*self.Vd_weights[i] + (1-beta1)*self.d_weights[i]
            self.Vd_biases[i] = beta1*self.Vd_biases[i] + (1-beta1)*self.d_biases[i]
            self.Sd_weights[i] = beta2*self.Sd_weights[i] + (1-beta2)*self.d_weights[i]**2
            self.Sd_biases[i] = beta2*self.Sd_biases[i] + (1-beta2)*self.d_biases[i]**2     
            
            vd_w_c = self.Vd_weights[i]/(1-beta1)
            vd_b_c = self.Vd_biases[i]/(1-beta1)
            sd_w_c = self.Sd_weights[i]/(1-beta2)
            sd_b_c = self.Sd_biases[i]/(1-beta2)    
            
            self.weights[i] = self.weights[i] - rate * (vd_w_c/(np.sqrt(sd_w_c) + self.epsilon))
            self.biases[i] = self.biases[i] - rate * (vd_b_c/(np.sqrt(sd_b_c) + self.epsilon))
                        
    def predict(self, p):
        p = (p - self.train_mean)/self.train_var
        self.layers = [p]
        self.layers.extend([[] for _ in self.size[1:]])
        for i in range(1, self.depth):
            z = np.dot(self.layers[i-1], self.weights[i-1]) + self.biases[i-1]
            self.layers[i] = self.activation(z, self.activations[i])
        return self.layers[self.depth-1]   
        
    def run_epoch(self, x, y, rate):                     
        self.forward_propagation(x)
        loss = self.compute_loss(y)
        self.backward_propagation(y)
        self.adam_optimizer(rate=0.001, beta1=0.9, beta2=0.999)
        return loss     
            
        


