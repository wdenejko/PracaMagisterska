import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time

start = time.time()
train = pd.read_csv('output.csv')
y = np.array(train.pop('label'))
x = np.array(train)/255.
plt.imshow(x[10].reshape(28,28), cmap='Greys', interpolation='nearest')


def plot_digit(pixels, label):
    img = pixels.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(label)
    plt.show()

split = 39000
x0 = x[:split]; x1 = x[split:]
y0 = y[:split]; y1 = y[split:]
mlp = MLPClassifier(solver='sgd', activation='relu',
                    hidden_layer_sizes=(100,30),
                    learning_rate_init=0.3, learning_rate='adaptive', alpha=0.1,
                    momentum=0.9, nesterovs_momentum=True,
                    tol=1e-4, max_iter=200,
                    shuffle=True, batch_size=300,
                    early_stopping = False, validation_fraction = 0.15,
                    verbose=True)
mlp.fit(x0,y0)
y_val = mlp.predict(x1)
accuracy = np.mean(y1 == y_val)
print accuracy

label = "Predicted: {0}, Actual: {1}".format(y1[100], y_val[100])
end = time.time()

plot_digit(x[100], label)
print(end - start)