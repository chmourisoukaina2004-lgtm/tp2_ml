import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,matthews_corrcoef,balanced_accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



#importer Dataset isis c'est un jeu de données des fleures 
import pandas as pd

url = "iris.csv"
df = pd.read_csv(url)
print(df.head())

#****************** 2.1 Compréhension des données ********************** : 
#idifier: variable cible : species 
        # variable features : sepal_length  sepal_width  petal_length  petal_width

#type de données : 
# print(df.dtypes)

#Statistiques descriptives (moyenne, médiane, etc.) 
# print(df.describe())

#Visualisations : 
#1 Histogrammes 
# df.hist(figsize=(10,8))
# plt.show()
#2 Boxplots 
# sns.boxenplot(data=df)
# plt.show()

#3 Corrélations : 
# correlations = df.corr(numeric_only=True)
# print(correlations)
# sns.heatmap(correlations,annot=True,cmap="coolwarm")
# plt.show()


#****************** 2.1 pretrairement  des données ********************** : 

#1 gestion de valeurs manquantes : 

#print(df.isnull().sum())
# - suppression de lignes : 
#df.dropna()

# - imputation de moy et mediane :
#df["age"].fillna(df["age"].mean(), inplace=True)

#df["age"].fillna(df["age"].median(), inplace=True)

# - mode
#df["sexe"].fillna(df["sexe"].mode()[0], inplace=True)

# - KNN

# b- Encodage des variables catégorielles 

# - label Encoding :
labelEncoded=LabelEncoder()
df["species_encoded"]=labelEncoded.fit_transform(df["species"])
print(df["species_encoded"])


# c - Normalisation / Standardisation 

# Standardisation
X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
y = df["species_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# d - Traitement des valeurs aberrantes (outliers) 
#IQR (Interquartile Range) 
# Q1 = df["sepal_length"].quantile(0.25)
# Q3 = df["sepal_length"].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1-1.5*IQR
# upper = Q3+1.5*IQR

# outliers_SL = df[(df["sepal_length"]>upper) | (df["sepal_length"] < lower)]
# print(f'outliers sepal_length : ',outliers_SL)


# Q1 = df["sepal_width"].quantile(0.25)
# Q3 = df["sepal_width"].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1-1.5*IQR
# upper = Q3+1.5*IQR

# outliers_SW = df[(df["sepal_width"]>upper) | (df["sepal_width"] < lower)]
# print(f'outliers sepal_width : ',outliers_SW)

# Q1 = df["petal_length"].quantile(0.25)
# Q3 = df["petal_length"].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1-1.5*IQR
# upper = Q3+1.5*IQR

# outliers_PL= df[(df["petal_length"]>upper) | (df["petal_length"] < lower)]
# print(f'outliers petal_length : ',outliers_PL)

# Q1 = df["petal_width"].quantile(0.25)
# Q3 = df["petal_width"].quantile(0.75)
# IQR = Q3 - Q1
# lower = Q1-1.5*IQR
# upper = Q3+1.5*IQR

# outliers_PW = df[(df["petal_width"]>upper) | (df["petal_width"] < lower)]
# print(f'outliers petal_width : ',outliers_PW)



#petal_length  petal_width
# pour voir les outliers : 
# sns.boxplot(x=df["sepal_length"])
# plt.show()
# sns.boxplot(x=df["sepal_width"])
# plt.show()
# sns.boxplot(x=df["petal_length"])
# plt.show()
# sns.boxplot(x=df["petal_width"])
# plt.show()

# => on remarque que seulement sepal_width qui contient des outliers , afin que on a une petite dataset donc on va utiliser TRANSFORMATION pour pas de perd de données : 

# TRANSFORMATION : 
# df["sepal_width"]=df["sepal_width"].clip(lower,upper)

# histograme pour voir apres la TRANSFORMATION de valeurs outliers :
# sns.boxplot(x=df["petal_width"])
# plt.title("apres TRANSFORMATION")
# plt.show()


#****************** 3 - Modèles à implémenter ********************** 
# Classification 
    # 1 - regression logistique : 
print("Modèles à implémenter")
print("1 - regression logistique")
# etape 1 : creation de modele vide
model = LogisticRegression()

# # etape 2 : entrainement :
model.fit(X_train_scaled,y_train)

# #etape 3 : prediction de valeur : 
y_pred = model.predict(X_test_scaled)

#etape 4 : affichage et comparison avec valeur reels : 
print("prediction de type fleurs : ",y_pred)
print("type fleurs reels : ",y_test.values)

#metriques d'evaluation 
#M1 : calcule accuracy de modele : 
accuracy=accuracy_score(y_test,y_pred)
print("accurcy de modele : ", accuracy)
#M2 : Precision : 
precision = precision_score(y_test,y_pred,average='weighted')
print("Precision :",precision)
#M3 : recall 
recal=recall_score(y_test,y_pred,average='weighted')
print('recall : ',recal)

#M4 : F1 score 
F1_scor=f1_score(y_test,y_pred,average='weighted')
print('f1 score : ',F1_scor)

#M5 : matrice de confusion : 
m_c=confusion_matrix(y_test,y_pred)


#m6 : balanced accuracy : 
balaced_acc=balanced_accuracy_score(y_test,y_pred)
print('balanced_accuracy_score :', balaced_acc)

#M7 : MCC 
mcc=matthews_corrcoef(y_test,y_pred)
print("mcc : ", mcc)

print("matrice de confusion : ", m_c)
sns.heatmap(
    m_c , annot=True , cmap="Blues"
)
plt.xlabel('prediction')
plt.ylabel('vrais valeurs')
plt.title('matrice de confusion RL')
plt.show()

    # 2- KNN 

# print("2 - KNN")
# # etape 1 : creation de modele vide
# KNN_modele = KNeighborsClassifier()

# # # etape 2 : entrainement : 
# # #  a- avant le l'entrainement on utulise grid search pour choisir le nombre optimale de voisins 'n_neighbors'
# param_grid = {
# "n_neighbors" : [1,3,5,7,9,11] #parametre de test
# }

# grid = GridSearchCV(
# KNN_modele , param_grid , cv=5 , scoring="accuracy"
# )
# # # b - entrainement : 
# grid.fit(X_train_scaled,y_train)
# # # c - meilleur model 
# print('meilleur K: ', grid.best_params_)

# # #etape 3 : prediction de valeur : 
# y_predit=grid.predict(X_test_scaled)

# # #etape 4 : affichage et comparison avec valeur reels : 
# print("prediction de type fleurs : ",y_predit)
# print("type fleurs reels : ",y_test.values)

# #m1  : calcule accuracy de modele : 
# accuracy=accuracy_score(y_test,y_predit)
# print("accurcy de modele : ", accuracy)

# #M2 : Precision : 
# precision = precision_score(y_test,y_predit,average='weighted')
# print("Precision :",precision)
# #M3 : recall 
# recal=recall_score(y_test,y_predit,average='weighted')
# print('recall : ',recal)

# #M4 : F1 score 
# F1_scor=f1_score(y_test,y_predit,average='weighted')
# print('f1 score : ',F1_scor)

# #M5 : matrice de confusion : 
# m_c=confusion_matrix(y_test,y_predit)
# print("matrice de confusion : ", m_c)

# #m6 : balanced accuracy : 
# balaced_acc=balanced_accuracy_score(y_test,y_predit)
# print('balanced_accuracy_score :', balaced_acc)

# #M7 : MCC 
# mcc=matthews_corrcoef(y_test,y_predit)
# print("mcc : ", mcc)

# sns.heatmap(
#     m_c , annot=True , cmap="Blues"
# )
# plt.xlabel('prediction')
# plt.ylabel('vrais valeurs')
# plt.title('matrice de confusion KNN')
# plt.show()


        # 3- Foret aleatoire : 
# print('3 - foret aleatoire : ')
# #creation de model : 
# FA_model=RandomForestClassifier(
#     n_estimators=100,
#     random_state=42
# )
# #entrainement 
# model=FA_model.fit(X_train_scaled,y_train)
# #prediction 
# y_predi=model.predict(X_test_scaled)
# # affichage :
# print('prediction de modele : ', y_predi)
# print("reel values : ",y_test.values)

#  #m1  : calcule accuracy de modele : 
# accuracy=accuracy_score(y_test,y_predi)
# print("accurcy de modele : ", accuracy)

# # #M2 : Precision : 
# precision = precision_score(y_test,y_predi,average='weighted')
# print("Precision :",precision)

# # #M3 : recall 
# recal=recall_score(y_test,y_predi,average='weighted')
# print('recall : ',recal)

# # #M4 : F1 score 
# F1_scor=f1_score(y_test,y_predi,average='weighted')
# print('f1 score : ',F1_scor)

# # #M5 : matrice de confusion : 
# m_c=confusion_matrix(y_test,y_predi)
# print("matrice de confusion : ", m_c)

# # #m6 : balanced accuracy : 
# balaced_acc=balanced_accuracy_score(y_test,y_predi)
# print('balanced_accuracy_score :', balaced_acc)

# # #M7 : MCC 
# mcc=matthews_corrcoef(y_test,y_predi)
# print("mcc : ", mcc)

# sns.heatmap(
#     m_c , annot=True , cmap="Blues"
# )
# plt.xlabel('prediction')
# plt.ylabel('vrais valeurs')
# plt.title('matrice de confusion FA')
# plt.show()

         # SVM 
#creation 
svm_model=SVC(kernel='linear')
#entrainement 
svm_model.fit(X_train_scaled,y_train)

#prediction 
y_svm=svm_model.predict(X_test_scaled)

#affichage : 
print("prediction : ",y_svm)
print('vrais values :', y_test.values)

#m1  : calcule accuracy de modele : 
acuracy=accuracy_score(y_test,y_svm)
print("accurcy de modele : ", acuracy)

# #M2 : Precision : 
precision = precision_score(y_test,y_svm,average='weighted')
print("Precision :",precision)

# #M3 : recall 
recal=recall_score(y_test,y_svm,average='weighted')
print('recall : ',recal)

# #M4 : F1 score 
F1_scor=f1_score(y_test,y_svm,average='weighted')
print('f1 score : ',F1_scor)

# #M5 : matrice de confusion : 
m_c=confusion_matrix(y_test,y_svm)
print("matrice de confusion : ", m_c)

# #m6 : balanced accuracy : 
balaced_acc=balanced_accuracy_score(y_test,y_svm)
print('balanced_accuracy_score :', balaced_acc)

# #M7 : MCC 
mcc=matthews_corrcoef(y_test,y_svm)
print("mcc : ", mcc)

sns.heatmap(
    m_c , annot=True , cmap="Reds"
)

plt.xlabel('prediction')
plt.ylabel('vrais valeurs')
plt.title('matrice de confusion - SVM')
plt.show()
