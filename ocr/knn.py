from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import time

start = time.time()
def plot_digit(pixels, label):
    img = pixels.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(label)
    plt.show()

labeled_images = pd.read_csv('output.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

knn = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree")
knn.fit(train_images, train_labels.values.ravel())
print knn.score(test_images,test_labels)


def test_prediction(index):
    predic = knn.predict(test_images.iloc[index:index+1])[0]
    actual = test_labels.iloc[index]['label']
    return (predic, actual)

index = random.randint(0, len(test_images)-1)
predic, actual = test_prediction(index)
end = time.time()
pixels = test_images.iloc[index].as_matrix()
label = "Predicted: {0}, Actual: {1}".format(predic, actual)

plot_digit(pixels, label)
print(end - start)