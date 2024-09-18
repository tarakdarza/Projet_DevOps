from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Charger le dataset Titanic
df = pd.read_csv('titanic.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Vérifier si les colonnes 'SibSp' et 'Parch' existent, sinon les créer avec des valeurs par défaut
if 'SibSp' not in df.columns:
    df['SibSp'] = 0
if 'Parch' not in df.columns:
    df['Parch'] = 0

# Exemple de modèle entraîné (logistic regression pour la survie sur Titanic)
X = df[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch']].dropna()
y = df['Survived'].dropna()

# Remplacer les valeurs manquantes
X['Age'].fillna(X['Age'].mean(), inplace=True)

# Entraîner le modèle (logistic regression)
model = LogisticRegression()
model.fit(X, y)

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour la page de sélection des lignes à afficher
@app.route('/select-dataset')
def select_dataset():
    return render_template('select_dataset.html')

# Route pour afficher les statistiques et un graphique
@app.route('/statistics')
def statistics():
    stats = df.describe().to_html(classes='table table-striped table-bordered table-hover')

    # Générer un graphique (exemple : distribution des âges)
    img_age = io.BytesIO()
    plt.figure(figsize=(6, 6))
    df['Age'].dropna().plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution des âges')
    plt.xlabel('Âge')
    plt.ylabel('Fréquence')
    plt.savefig(img_age, format='png')
    img_age.seek(0)
    plot_url_age = base64.b64encode(img_age.getvalue()).decode('utf8')

    return render_template('statistics.html', stats=stats, plot_url_age=plot_url_age)

# Route pour afficher le dataset avec le nombre de lignes sélectionné
@app.route('/dataset', methods=['GET'])
def dataset():
    num_rows = request.args.get('num_rows', default='5', type=str)

    if num_rows == 'all':
        table = df
    else:
        try:
            num_rows = int(num_rows)
            table = df.head(num_rows)
        except ValueError:
            table = df.head(5)

    return render_template('dataset.html', tables=[table], titles=table.columns.values)

# Route pour afficher le formulaire de prédiction
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Route pour gérer la prédiction après soumission du formulaire
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    pclass = int(request.form['pclass'])
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    fare = float(request.form['fare'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])

    # Créer un array avec ces valeurs
    input_data = np.array([[pclass, age, sex, fare, sibsp, parch]])

    # Faire la prédiction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Survécu"
    else:
        result = "Non survécu"

    # Renvoyer le résultat à une page de résultats
    return render_template('result.html', result=result)

# Ajouter un gestionnaire d'erreurs pour 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', error_code=404, message=str(e)), 404

if __name__ == '__main__':
    app.run(debug=False)
