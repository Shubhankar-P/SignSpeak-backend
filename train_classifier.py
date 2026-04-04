# I am using mediapipe as a hand detector and landmark detector and a Random Forest classifier as sign classifier.


import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labelsx
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Flatten the data and ensure landmarks are structured as arrays
data_flattened = np.array(data)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_flattened, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
