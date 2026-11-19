import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('drug.csv')
print(data.head())

le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()
le_drug = LabelEncoder()

data['Sex'] = le_sex.fit_transform(data['Sex'])
data['BP'] = le_bp.fit_transform(data['BP'])
data['Cholesterol'] = le_cholesterol.fit_transform(data['Cholesterol'])
data['Drug'] = le_drug.fit_transform(data['Drug'])

print(data.head())

X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted Values:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree:", accuracy)
print("Accuracy in Percentage:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print(classification_report(y_test, y_pred, target_names=le_drug.classes_))

new_patient = pd.DataFrame({
    'Age': [30],
    'Sex': le_sex.transform(['F']),
    'BP': le_bp.transform(['HIGH']),
    'Cholesterol': le_cholesterol.transform(['NORMAL']),
    'Na_to_K': [15.5]
})

prediction = model.predict(new_patient)
print("Predicted Drug:", le_drug.inverse_transform(prediction)[0])
