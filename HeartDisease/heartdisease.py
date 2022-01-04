import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
encoder = LabelEncoder()

data=pd.read_csv('heart.csv')

encoder.fit(data['ChestPainType'])
data['ChestPainType_encoded'] = encoder.transform(data['ChestPainType'])
encoder.fit(data['Sex'])
data['Sex_encoded'] = encoder.transform(data['Sex'])
encoder.fit(data['ST_Slope'])
data['ST_Slope_encoded'] = encoder.transform(data['ST_Slope'])
encoder.fit(data['RestingECG'])
data['RestingECG_encoded'] = encoder.transform(data['RestingECG'])
encoder.fit(data['ExerciseAngina'])
data['ExerciseAngina_encoded'] = encoder.transform(data['ExerciseAngina'])

x=data.drop(columns=['HeartDisease','ChestPainType','Sex','ST_Slope','RestingECG','ExerciseAngina'])

x['RestingBP']=((x['RestingBP']-x['RestingBP'].min())/(x['RestingBP'].max()-x['RestingBP'].min()))
x['Cholesterol']=((x['Cholesterol']-x['Cholesterol'].min())/(x['Cholesterol'].max()-x['Cholesterol'].min()))*100
x['MaxHR']=((x['MaxHR']-x['MaxHR'].min())/(x['MaxHR'].max()-x['MaxHR'].min()))
#x['Oldpeak']=((x['Oldpeak']-x['Oldpeak'].min())/(x['Oldpeak'].max()-x.min()))*

y=data['HeartDisease']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(200, input_shape=(None,11), activation='sigmoid')) 
model.add(tf.keras.layers.Dense(400, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

heart_disease=model.fit(x_train,y_train, epochs=40)
model.summary()
model.evaluate(x_test,y_test)

plt.plot(heart_disease.history['loss'], label='MAE (training data)')
plt.plot(heart_disease.history['accuracy'], label='MAE (validation data)')
plt.legend(['loss', 'accuracy'])
plt.xlabel('epoch')
plt.show()
