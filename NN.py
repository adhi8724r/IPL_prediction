import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("DataSet//ipl_dataframe.csv")

# Define features and target
X = data.drop(['result'], axis=1)
y = data[['result']]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Apply transformation
X_transformed = preprocessor.fit_transform(X)

# Convert target variable to numerical (if not already encoded)
if y.nunique()[0] > 2:
    encoder = OneHotEncoder(sparse=False)
    y_transformed = encoder.fit_transform(y)
else:
    y_transformed = y.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.3, random_state=43)

# Build Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(y_train.shape[1], activation='softmax' if y_train.shape[1] > 1 else 'sigmoid')
])

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy' if y_train.shape[1] > 1 else 'binary_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
