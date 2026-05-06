
# PARTIE 1 : *****************************************************************
# importer la biblio qui lire les données : 
# dans cmd telecharger pandas : pip install pandas
import pandas as pd

# importer dataset : 
url="./titanic.csv"
dataset=pd.read_csv(url)

# les types de données : 
print(df.dtypes)

# les premieres lignes : 
print(dataset.head())

# Vérification des valeurs manquantes par colonne
valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)

# PARTIE 2 : *****************************************************************
# 1. Gestion des valeurs manquantes :

# 1. Imputation des variables numériques (Age) par la médiane
# On utilise la médiane car elle est moins sensible aux valeurs extrêmes que la moyenne
df['Age'] = df['Age'].fillna(df['Age'].median())

# 2. Imputation des variables catégorielles (Embarked) par le mode
# .mode() retourne une Série, on prend donc l'indice [0]
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)


# 3. Suppression des colonnes trop vides (> 40%)
# La colonne 'Cabin' a généralement plus de 70% de valeurs manquantes
df.drop(columns=['Cabin'], inplace=True, errors='ignore')

# 2. Gestion des valeurs aberrantes :

import seaborn as sns
import matplotlib.pyplot as plt

# Visualisation avec Seaborn
sns.boxplot(x=df['Fare'])
plt.title("Détection des valeurs aberrantes pour le tarif (Fare)")
plt.show()

# Détecter les valeurs aberrantes (outliers)
# Boxplot
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Age'])
plt.show()

# PARTIE 3  : *****************************************************************
# 1. Encodage des variables catégorielles :

    # 1. IDENTIFICATION DES COLONNES CATÉGORIELLES
        # On cherche les colonnes de type 'object' (texte)
        cols_categoriques = df.select_dtypes(include=['object']).columns
        print("Colonnes à encoder :", list(cols_categoriques))

    # 2. Appliquez l'encodage one-hot pour les variables nominales (par exemple, "Sexe").
        # ENCODAGE ONE-HOT (Variables nominales sans ordre)
        # On transforme 'Sex' et 'Embarked' en colonnes binaires (0 ou 1)
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # 3. Pour les variables ordinales (par exemple, "Classe"), vous pouvez les encoder numériquement (1 pour "1ère classe", 2 pour "2ème classe", etc.).
        # Définir l'ordre logique dans un dictionnaire
        mapping_ordre = {
            "First": 1, 
            "Second": 2, 
            "Third": 3
        }
        # Appliquer la transformation
        df['Pclass'] = df['Pclass'].map(mapping_ordre)  
        # La fonction .map() : Elle regarde chaque ligne de la colonne, cherche le mot dans votre dictionnaire, 
        # et le remplace instantanément par le chiffre associé. Si un mot n'est pas dans le dictionnaire,
        # il devient NaN (vide), ce qui vous permet aussi de repérer des erreurs de saisie.
        print(df[['Pclass']].head()) 

# 2. Normalisation et Standardisation des données :
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    # etape 1 : SÉLECTION DES COLONNES NUMÉRIQUES
    # On travaille sur l'Age et le Fare (Prix du billet)
    cols_numeriques = ['Age', 'Fare']

    # 1. NORMALISATION (Min-Max Scaling) -> Intervalle [0, 1]
    scaler_minmax = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[cols_numeriques] = scaler_minmax.fit_transform(df[cols_numeriques])

    # 2. STANDARDISATION (Z-score) -> Moyenne=0, Écart-type=1
    scaler_std = StandardScaler()
    df_standardized = df.copy()
    df_standardized[cols_numeriques] = scaler_std.fit_transform(df[cols_numeriques])

    # 3. COMPARAISON DES DISTRIBUTIONS (Visualisation)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Histogramme Avant Transformation (Original) ---
    sns.histplot(df['Age'], kde=True, ax=axes[0], color='blue')
    axes[0].set_title(f"Original (Age)\nMin: {df['Age'].min()}, Max: {df['Age'].max()}")

    # --- Histogramme Après Normalisation ---
    sns.histplot(df_normalized['Age'], kde=True, ax=axes[1], color='green')
    axes[1].set_title("Normalisation (Min-Max)\nValeurs entre [0, 1]")

    # --- Histogramme Après Standardisation ---
    sns.histplot(df_standardized['Age'], kde=True, ax=axes[2], color='red')
    axes[2].set_title("Standardisation (Z-score)\nMoyenne = 0")

    plt.tight_layout()
    plt.show()

