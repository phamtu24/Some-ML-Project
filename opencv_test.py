import cv2
import numpy as np
import matplotlib.pyplot as plt 

train_data = np.random.randint(0, 100, (25, 2)).astype(np.float32)
value = np.random.randint(0, 2, (25, 1)).astype(np.float32)
test = np.random.randint(0, 100, (1, 2)).astype(np.float32)

red = train_data[value.ravel()==1]
blue = train_data[value.ravel()==0]
plt.scatter(red[:,0], red[:,1], 100, 'r', 's')
plt.scatter(blue[:,0], blue[:,1], 100, 'b', '^')
plt.scatter(test[:,0], test[:,1], 100, 'g', 'o')
knn = cv2.ml.KNearest_create()
knn.train(train_data, 0, value)
temp, result, neighbor, distance  = knn.findNearest(test, 3)
print(result, neighbor, distance)
plt.show()
