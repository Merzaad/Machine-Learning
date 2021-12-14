import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
y=data['HeartDisease']
print(y.shape)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=(None, 763, 8), activation='sigmoid')) 
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.fit(x_train,y_train, epochs=100 )
model.evaluate(x_test,y_test)
