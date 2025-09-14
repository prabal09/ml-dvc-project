**Titanic Survival Prediction with DVC**



Project Objective



* Download Titanic dataset.
* Build a simple ML model (e.g., logistic regression).
* Use DVC to version the dataset and the trained model.
* Track metrics across experiments (like accuracy, F1 score). (Idea: simulate "data drift" by slightly modifying the data and retraining.)





Creating the dvc.yaml
dvc stage add -n train -d data/titanic.csv -d src/train.py -o models/titanic\_logistic\_regression\_model.pkl -M metrics\_logistic\_regression.json python src/train.py

