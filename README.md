# Aleatoric-Uncertainty-Visualizer
Diese Python-Projekt wurde im Rahmen der Bachelorarbeit "Aleatorische Unsicherheit bei Neuronalen Netzen: Eine Analyse von Output-Verteilungen" entwickelt. Die Skripte ermöglichen die Analyse von aleatorischer Unsicherheit in neuronalen Netzen durch die Verarbeitung von drei ausgewählten Datensätzen: [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris), [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) und [Melbourne Housing Dataset](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot/data).

Der Ordner ```legacy``` enthält die in der Bachelorarbeit verwendeten Skripte, inklusive ```imputation_deterministic.py``` und ```imputation_eval_main_V2.py```. Die restlichen Skripte fassen den Code-Ablauf verständlicher und generischer zusammen.

Ablauf
1. Imputation der Daten
2. Visualisierung der Kernel Density Estimation (KDE) der einzelnen Attribute
3. Erstellen eines Neuronalen Netzes (NN)
4. Erstellen einer Monte Carlo Simulation (MCS)
5. Erstellen eines Gaussian Mixture Model (GMM)
6. Visualisierung des GMM

Das Python-Skript ermöglicht die Analyse aleatorischer Unsicherheiten in neuronalen Netzen durch folgende Schritte:

## Imputation der Daten

Die Imputation der Daten erfolgt auf zwei Arten. Dies ermöglicht den Vergleich der stochastischen Impuation mittels KDE Verteilungen und der deterministischen Impution, bei der der Mean des Attributes für fehlende Werte verwendet wird. Zunächst wird ein Prozentsatz der Daten (für die Bachelorarbeit: 30%) zufällig aus bestimmten Attributen gelöscht. Die festgelegten Attribute können somit als unsicher betrachtet werden. Im nächsten Schritt imputieren die beiden Arten diese fehlenden Werte wieder. 

## Visualisierung der KDE

Für einen Vergleich wird eine KDE für die Attribute aus den imputierten Daten erstellt, sowie aus den Ground Truth Daten.
<div style="display: flex;">
  <img src="/images/iris_kde_pw_gt.png" style="width: 45%;" />
  <img src="/images/iris_kde_pw.png" style="width: 45%;" /> 
</div>

## Erstellen eines NN

Für die drei Datensätze, wurden jeweils unterschiedliche Netz-Architekturen verwendet, die für gute Ergebnisse gesorgt habe.
- Iris: 2 Relu-Schichten (hidden), Softmax-Schicht (Output)
- Heart Disease: 2 Relu-Schichten (hidden), Sigmoid-Schicht (Output)
- Melbourne Housing: 4 Relu-Schichten (hidden), Linear-Schicht (Output)

## Erstellen einer MCS

Die MCS erstellt eine bestimmte Anzahl an Vorhersagen des Modells (für die Bachelorarbeit: 1000). Außerdem werden die Vorhersagen der vorletzten Schicht für die Datensätze Iris und Heart Disease gespeichert, für eine bessere Untersuchung.

## Erstellen eines GMM

Hierfür wurde zunächst mittels Bayesian Information Criterion (BIC) eine passende Anzahl an Komponenten für das GMM bestimmt. Das GMM wurde dann für die Datensätze Iris und Heart Disease auf der vorletzten Schicht gefittet und für den Melbourne Housing Datensatz auf der letzten Schicht.

## Visualisierung des GMM

Mittels Plots wie Scatter und Surface wurde das jeweilige GMM mit dessen Komponenten visualisiert.
<div style="display: flex;">
  <img src="/images/iris_gmm_gt_final.png" style="width: 45%;" />
  <img src="/images/iris_gmm_final.png" style="width: 45%;" /> 
</div>
<div style="display: flex;">
  <img src="/images/iris_pdf.png" style="width: 30%;" />
  <img src="/images/iris_log.png" style="width: 30%;" /> 
  <img src="/images/iris_scatter_1-4-3_neuron.png" style="width: 30%;" /> 
</div>

## Verwendung des Skripts
```bash
pip install requirements.txt

python visualizer.py --dataset=iris --num_sim=1000 --missing=30 --use_gt=False
```
- --dataset: gibt den Datensatz an (```iris``` ```heart``` ```housing```) (default: iris)
- --num_sim: gibt die Anzahl an Monte Carlo Simulation an (default: 1000)
- --missing: gibt den Prozentsatz an fehlenden Werten an (default: 30)
- --use_gt: gibt an, ob die Originaldaten verwendet werden sollen (default: False)
