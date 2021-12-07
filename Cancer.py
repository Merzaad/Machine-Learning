
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
data=pd.read_csv('Cancer.csv')
pred=pd.read_csv('CancerPred.csv')
x=data.drop(columns=['diagnosis(1=m, 0=b)'])
y=data['diagnosis(1=m, 0=b)']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=(455, 30), activation='sigmoid')) #x_train.shape
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#binary_crossentropy because of y ( 0 or 1 )

model.fit(x_train,y_train, epochs=100 )
model.evaluate(x_test,y_test)

prediction = model.predict(pred)
print(pd.DataFrame(prediction))
