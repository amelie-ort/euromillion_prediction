import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class Model():
    def __init__(self, data = pd.read_csv("EuroMillions_numbers.csv", sep=';')):
        
        self.data_formatting(data)

        self.create_model()

    def data_formatting(self, data):
        formatted_data = data[["N1", "N2", "N3", "N4", "N5", "E1", "E2"]].assign(HasWin = np.ones(data.shape[0], dtype=int))
        false_data_np = np.hstack((np.random.randint(50, size=(4*data.shape[0], 5)), np.random.randint(12, size=(4*data.shape[0], 2)), np.zeros((1, 4*data.shape[0]), dtype=int).T))
        false_data = pd.DataFrame(
            data=false_data_np,
            columns=["N1", "N2", "N3", "N4", "N5", "E1", "E2", "HasWin"],
            index=list(range(data.shape[0], data.shape[0] + false_data_np.shape[0]))
        )
        total_data = pd.concat([formatted_data, false_data])

        self.train_data, self.test_data = train_test_split(total_data, test_size=0.2)

    def create_model(self): 
        x = self.train_data.drop('HasWin', 1)
        y = self.train_data['HasWin'].astype('category').cat.codes

        self.modele_logit = LogisticRegression(solver="newton-cg")
        self.modele_logit.fit(x,y)

    def test(self):
        error = np.mean(self.modele_logit.predict(self.test_data.drop('HasWin', 1)) != self.test_data['HasWin'].to_numpy())
        return "Le modèle fonctionne dans " + str(int((1 - error) * 100)) + "% des cas"