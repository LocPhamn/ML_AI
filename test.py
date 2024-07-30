import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report ,r2_score

# Tải dữ liệu Boston Housing
data = pd.read_csv(r'D:\Python plus\diabetes.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
target = "Outcome"
x = data.drop(target,axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Xây dựng mô hình
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])

# Hiển thị cấu trúc mô hình

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)
# Đánh giá mô hình
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')
# Dự đoán giá nhà
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
x_tt = np.arange(0,len(X_test))
fx,ax = plt.subplots()
ax.plot(x_tt[:int(0.8*len(x_tt))],y_test, label="giá trị thực")
ax.plot(x_tt[int(0.8*len(x_tt)):],predictions, label="giá trị dự đoán")
ax.legend()
ax.set_xlabel("date")
ax.set_ylabel("users")
plt.tight_layout()
plt.show()
# Hiển thị một số kết quả dự đoán
print(classification_report(y_test,predictions))
