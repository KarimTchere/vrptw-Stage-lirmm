

# 🚚 VRPTW Solver  

## 📌 Description  

Cette application **Streamlit** permet de résoudre le **problème de tournée de véhicules avec fenêtres temporelles (VRPTW)**.  
Elle offre plusieurs fonctionnalités, notamment :  

✔️ **Optimisation des tournées de véhicules**  
✔️ **Visualisation des itinéraires**  
✔️ **Analyse comparative avant et après suppression de clients**  

---

## ⚡ Fonctionnalités  

### 🏆 **VRPTW Solver V1**  
- 🔹 Résolution du problème VRPTW et affichage des résultats détaillés.  
- 📥 **Input requis** : Instance de Solomon au format `.txt`.  

### 📉 **VRPTW Solver V2**  
- 🔹 Simulation de la réduction du nombre de clients et affichage des résultats.  
- 📥 **Input requis** : Instance de Solomon au format `.txt`.  

### 🔍 **Tournées Similarity**  
- 🔹 Comparaison des tournées avant et après suppression de clients.  
- 🔹 Utilisation de l'indice de similarité de **Jaccard**.  
- 📥 **Input requis** : **Output de VRPTW Solver V2 uniquement** ⚠️  

---

## 🛠️ Installation  

Pour exécuter cette application, vous devez avoir **Python** installé ainsi que les dépendances suivantes :  

```r
- Streamlit  
- Hexaly Optimizer  
- Pandas  
- Matplotlib  
- Plotly  
- VRPLib  
- NumPy  

⚠️  TRES IMPORTANT : avoir une licence hexaly !!! ⚠️  

Pour lancer l'application,  ouvrez un terminal, naviguez jusqu'au dossier source, puis exécutez la commande suivante :

streamlit run nom_du_fichier.py

