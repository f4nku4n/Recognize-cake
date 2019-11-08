import numpy as np
import cv2 as cv
import os
from skimage.feature import hog
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import joblib  # save / load model
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

basepath = 'train/'

i = 0
labels = open("labels.txt","w") # Tạo từ điển các nhãn
flag = False
X_train = np.array([[]])
Y_train = []
X_test = np.array([[]])
Y_test = []

for entry in os.listdir(basepath):  # Duyệt các folder chứa ảnh
    path = os.path.join(basepath, entry)  # Tạo đường dẫn
    feature = []  # Tạo danh sách các vector đặc trưng
    label = []  # Tạo danh sách các nhãn của từng ảnh
    for entries in os.listdir(path):  # Duyệt từng ảnh
        if os.path.isfile(os.path.join(path, entries)):
            image = cv.imread(os.path.join(path, entries))  # Đọc ảnh
            image = cv.resize(image, (256, 256))  # Resize tấm ảnh thành kích thước 256 x 256
            fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), multichannel=True)  # Dùng HOG trích xuất ra
            # vector đặc trưng
            feature.append(fd)  # Thêm vector đặc trưng vào danh sách các vector đặc trưng
            label.append(i)  # Thêm nhãn tương ứng vào danh sách các nhãn
    # Chia dữ liệu của loại bánh tương ứng thành các bộ train và test (tỉ lệ 7:3)
    x_train, x_test, y_train, y_test = train_test_split(np.array(feature), label, test_size=0.3,
                                                        random_state=np.random)
    # Tạo bộ train và test tổng
    if flag is False:
        X_train = x_train
        X_test = x_test
        flag = True
    else:
        X_train = np.concatenate((X_train, x_train), axis=0)
        X_test = np.concatenate((X_test, x_test), axis=0)
    Y_train.extend(y_train)
    Y_test.extend(y_test)
    labels.write("{0} {1}\n".format(i,entry))
    i += 1

model = LogisticRegression()
model = model.fit(X_train, Y_train)
model_file = joblib.dump(model, "clf_logistic.joblib")


print("PREDICT TRAINING SET")
predict_label_train = model.predict(X_train)
print("Accruracy:", accuracy_score(Y_train, predict_label_train))
print("Matrix confusion:")
print(metrics.confusion_matrix(Y_train, predict_label_train))
print("Report classification:")
print(metrics.classification_report(Y_train, predict_label_train, digits=3))

print("PREDICT TESTING SET")
predict_label_test = model.predict(X_test)
print("Accruracy:", accuracy_score(Y_test, predict_label_test))
print("Matrix confusion:")
print(metrics.confusion_matrix(Y_test, predict_label_test))
print("Report classification:")
print(metrics.classification_report(Y_test, predict_label_test, digits=3))
