Vérifie que le fichier model.pkl est généré :

Le fichier model.pkl doit être généré en exécutant model.py,
 qui entraîne le modèle et le sauvegarde sous forme de fichier pickle.
Exécuter le script model.py pour générer model.pkl :

Ouvre un terminal dans Visual Studio Code (dans le répertoire de ton projet) 
et exécute le script model.py pour entraîner le modèle et sauvegarder le fichier pickle :

python model.py

Cela va générer le fichier model.pkl dans ton répertoire actuel.
Assure-toi que model.pkl est bien présent :

Après avoir exécuté model.py, vérifie que le fichier model.pkl a bien été créé dans ton répertoire de projet.
Si le fichier est correctement généré, relance ton application Flask :

python app.py

Résumé des actions :
Exécute le fichier model.py pour générer model.pkl.
Vérifie que le fichier model.pkl est dans le même répertoire que app.py.
Lance à nouveau app.py pour démarrer l'application Flask.