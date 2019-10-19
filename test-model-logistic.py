import matplotlib.pyplot as plt
import joblib  # save / load model
import numpy as np
import cv2 as cv
from skimage.feature import hog

model_svm = joblib.load("clf_logistic.joblib")
label_dict = open("labels.txt")
label_dict = label_dict.readlines()
dict = {}
for x in label_dict:
    i = x.split(" ")
    dict[int(i[0])] = i[1][:-1]
print("Nhap duong dan cua hinh: ",end='')
path = input()
image = cv.imread(path)
image = cv.resize(image, (256, 256))
fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), multichannel=True)
predict_svm = model_svm.predict(np.array([fd]))
cv.putText(image, dict[predict_svm[0]], (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
cv.putText(image, "Logistic.R", (0, 250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()