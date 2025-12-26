#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
data = pd.read_csv("student_data.csv")
def performance_label(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"
data["performance"] = data["final_score"].apply(performance_label)
X = data[["attendance", "study_hours", "assignment_score","previous_score", "participation"]]
y = data["performance"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
plt.figure()
plt.hist(data["final_score"], bins=20)
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Academic Scores")
plt.show()
new_student = np.array([[85, 12, 78, 72, 8]])
new_student = scaler.transform(new_student)
prediction = model.predict(new_student)
predicted_label = label_encoder.inverse_transform(prediction)
print("\nPredicted Academic Performance:", predicted_label[0])
if predicted_label[0] == "Low":
    print("Academic Risk Detected")
else:
    print("No Academic Risk")


# In[ ]:





# In[ ]:




