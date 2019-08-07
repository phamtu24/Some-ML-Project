import numpy as np
import matplotlib.pyplot as plt 
import cv2 

img = cv2.imread('digits.png', 0)
test_img = cv2.imread('so2.png', 0)
test_img = cv2.resize(test_img, (20, 20))


cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)]
cells = np.array(cells)
test_img = np.array(test_img)
test_img = test_img.reshape(-1, 400).astype(np.float32)
#training array
train = cells[:,:25].reshape(-1, 400).astype(np.float32)
#testing array
test = cells[:,25:50].reshape(-1, 400).astype(np.float32)
#label array
k = np.arange(10)
train_labels = np.repeat(k, 125)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, 0, train_labels)
temp, result, neighbor, distance = knn.findNearest(test_img, 5)
print(result)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()