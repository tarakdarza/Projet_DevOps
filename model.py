import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Générer des données fictives pour l'exemple
data = {
    'age': [25, 45, 35, 50, 23, 30, 40],
    'bmi': [18.5, 24.0, 27.5, 30.0, 22.5, 25.0, 26.5],
    'smoker': [0, 1, 1, 0, 0, 1, 1],  # 0: Non fumeur, 1: Fumeur
    'risk': [0, 1, 1, 1, 0, 0, 1]  # 0: Pas de risque, 1: Risque élevé
}

df = pd.DataFrame(data)

# Préparer les données
X = df[['age', 'bmi', 'smoker']]
y = df['risk']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné avec pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
