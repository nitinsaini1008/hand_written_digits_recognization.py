import numpy
import pandas
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
#plt.matshow(x_train[0])
#plt.show()
#print(x_train[0])
x_test=x_test/255
x_train=x_train/255
model=Sequential()
model.add(Flatten(input_shape=[28,28]))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
#print(y_train.shape)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train)
y_ob=model.predict(x_test)
print(numpy.argmax(y_ob[1]))
plt.matshow(x_test[1])
plt.show()
print(model.evaluate(x_test,y_test))
