import numpy as np
from sklearn.model_selection import train_test_split

class Layer:
    def __init__(self, n_inputs, n_neurons, init_method="random", activation="linear", learning_rate=0.5):
        self.weights = None
        self.initParam(n_inputs, n_neurons, init_method)    # initiating weights and biases
        self.biases = np.zeros((1, n_neurons))
        self.activation_function = activation
        self.learning_rate = learning_rate
        self.inputs = None
        self.Z = None
        self.activation = None
        self.output = None
        self.next_layer = None
        self.dZ = None
        self.dW = None
        self.db = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.Z = np.dot(inputs, self.weights) + self.biases
        if(self.activation_function == "relu"):
            self.activation = Activation_RelU(self.Z)
        elif(self.activation_function == "sigmoid"):
            self.activation = Activation_Sigmoid(self.Z)
        elif(self.activation_function == "linear"):
            self.activation = Activation_Linear(self.Z)
        elif(self.activation_function == "softmax"):
            self.activation = Activation_Softmax(self.Z)
        else:
            raise ValueError("Invalid layer activation")
        
        self.output = self.activation.forward()
        
    def backward(self, y_true):
        m = y_true.shape[0]
        if(self.next_layer == None):
            if(len(y_true)==0):
                raise ValueError("Invalid target.")
            one_hot_y = oneHotEncode(y_true)
            self.dZ = self.output - one_hot_y
        else:
            layer2 = self.next_layer
            self.dZ = (np.dot(layer2.weights, layer2.dZ.T) * self.activation.derivation().T).T
        self.dW = 1 / m * np.dot(self.dZ.T, self.inputs)
        self.db = 1 / m * np.sum(self.dZ, 0)
    
    def updateParam(self):
        self.weights = self.weights - self.learning_rate * self.dW.T
        self.biases = self.biases - self.learning_rate * self.db
    
    def initParam(self, n_inputs, n_neurons, method):
        if(method == "random"):
            self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        elif(method == "xavier"):
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / (n_inputs + n_neurons))
        elif(method == "he"):
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        elif(method == "orthogonal"):
            W = np.random.randn(n_inputs, n_neurons)
            Q, _ = np.linalg.qr(W)
            self.weights = Q
        elif(method == "glorot"):
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / (n_inputs + n_neurons))
        elif(method == "lecun"):
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        else:
            raise ValueError("Invalid params initialization method.")
    
    def setLearningRate(self, a):
        self.learning_rate = a
    
    def setNextLayer(self, next_layer):
        self.next_layer = next_layer
        
class Activation_Linear:
    def __init__(self, inputs):
        self.Z = inputs
        
    def forward(self):
        self.output = self.Z
        return self.output
        
    def derivation(self):
        return 1
        
class Activation_RelU:
    def __init__(self, inputs):
        self.Z = inputs
    
    def forward(self):
        self.output = np.maximum(0, self.Z)
        return self.output
        
    def derivation(self):
        return np.where(self.Z > 0, 1, 0)
    
class Activation_Softmax:
    def __init__(self, inputs):
        self.Z = inputs
    
    def forward(self):
        exps = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))
        exps_sum = np.sum(exps, axis=1, keepdims=True)
        self.output = exps / exps_sum
        return self.output
    
    def derivative(self):
        return np.diag(self.output) - np.outer(self.output, self.output)
        
class Activation_Sigmoid:
    def __init__(self, inputs):
        self.Z = inputs
        
    def forward(self):
        self.output = 1 / (1 + np.exp(-self.Z))
        return self.output
        
    def derivative(self):
        return self.output * (1 - self.output)
        
class Regularization:
    def __init__(self, weights, regularization="", lambda_=1):
        self.regularization = regularization
        self. weights = weights
        self.lambda_ = lambda_
    
    def calculate(self):
        if(self.regularization == "l1"):
            reg = l1_regularization(self.weights, self.lambda_)
        elif(self.regularization == "l2"):
            reg = l2_regularization(self.weights, self.lambda_)
        elif(self.regularization == ""):
            reg = 0
        else:
            raise ValueError('Regularization must either be "l1", "l2", or "".')
        return reg

def l1_regularization(weights, lambda_):
  return lambda_ * np.sum(np.abs(weights))

def l2_regularization(weights, lambda_):
  return 0.5 * lambda_ * np.sum(np.square(weights))

class Loss_MeanSquared:
    def __init__(self, weights=[], regularization="", lambda_=1):
        self.regularization = regularization
        self. weights = weights
        self.lambda_ = lambda_
              
    def calculate(self, y_pred, y_true):
        squared = [(1-y_pred[i,y])**2 + np.sum(np.square(np.array(y_pred[i,:y]+y_pred[i,y+1:]))) for i, y in enumerate(y_true)]
        loss = np.mean(squared)
        reg = Regularization(weights=self.weights, regularization=self.regularization, lambda_=self.lambda_)
        loss += reg.calculate()
        return loss
    
class Loss_CrossEntropy:
    def __init__(self, weights=[], regularization="", lambda_=1):
        self.regularization = regularization
        self. weights = weights
        self.lambda_ = lambda_
        
    def calculate(self, y_pred, y_true):
        n_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if(len(y_true.shape) == 1): # number categorical: 0, 1, 2
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        elif(len(y_true.shape) == 2):   # one-hot encoded target label
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        loss = np.mean(-np.log(correct_confidences))
        reg = Regularization(weights=self.weights, regularization=self.regularization, lambda_=self.lambda_)
        loss += reg.calculate()
        return loss
    
def oneHotEncode(y):
    one_hot_y = np.zeros((y.size, y.max()+1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y


class ANN:
    def __init__(self, X, y, n_neurons=16, init_method="random", loss="cross_entropy", regularization="", lambda_=0.1):
        self.X = X.values
        self.y = y.values
        self.one_hot_y = oneHotEncode(self.y)
        self.n_neurons = n_neurons
        self.init_method = init_method
        self.loss = loss
        self.regularization = regularization
        self.lambda_ = lambda_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
    def gradientDescent(self, n_epoch=100, learning_rate=0.1, batch_size=10):
        # Contoh susunan dengan 2 hidden layer
        layer1 = Layer(self.X.shape[1], self.n_neurons, activation="relu", learning_rate=learning_rate, init_method=self.init_method)
        layer2 = Layer(self.n_neurons, len(np.unique(self.y)), activation="softmax", learning_rate=learning_rate, init_method=self.init_method)
        layer1.setNextLayer(layer2)
        
        for i in range(1,n_epoch+1):
            layer1.forward(self.X)
            layer2.forward(layer1.output)
            
            self.predict(layer2.output)
            accuracy = self.getAccuracy()
            loss = self.getLoss(layer2.output, self.loss)
            
            layer2.backward(self.y) # one hot encoding included in backward
            layer1.backward(self.y)
            
            layer1.updateParam()
            layer2.updateParam()
            if(i%batch_size == 0):
                print("Epoch:",i,"\t\t","Accuracy:",accuracy,"\t\t","Loss:",loss)
     
    def predict(self, last_layer_output):
        self.prediction = np.argmax(last_layer_output, 1)
        return self.prediction
    
    def getAccuracy(self):
        return np.sum(self.prediction == self.y) / self.y.size  
    
    def getLoss(self, last_layer_output, method):
        if(method == "cross_entropy"):
            loss = Loss_CrossEntropy().calculate(last_layer_output, self.one_hot_y)
        elif(method == "mean_square"):
            loss = Loss_MeanSquared().calculate(last_layer_output, self.one_hot_y)
        else:
            raise ValueError("Invalid loss function.")
        return loss
    
# import pandas as pd
# # df = pd.read_csv("dataset/tes.csv")
# df = pd.read_csv("dataset/SleepyDriverEEGBrainwave.csv")
# y = df["classification"]
# X = df.drop("classification", axis=1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# model = ANN(X, y, n_neurons=5, init_method="orthogonal")
# model.gradientDescent(n_epoch=200)