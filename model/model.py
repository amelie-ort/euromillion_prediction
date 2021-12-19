import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

class IAModel:

    """class containing the prediction model"""
   
    def __init__(self):
        self.data_manager = DataManager()
        self.model = self.read_model() if os.path.isfile('./model.txt') else self.genere_model()
           
    def genere_model(self):
        """this function generates the prediction model"""
        self.data_manager.inject_false_data()
        return self.train()
   
    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data_manager.data.drop('HasWin', 1), self.data_manager.data['HasWin'], test_size=0.2)    
        params = {
            'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
            'solver' : ['lbfgs', 'liblinear', 'sag', 'saga']
        }  
        grid = GridSearchCV(LogisticRegression(), params, cv=5)
        grid.fit(x_train, y_train)
        model = grid.best_estimator_
       
        f = open("score.txt", "w")
        f.write(str(model.score(x_test, y_test)))
        f.close()

        f = open("model.txt", "wb")
        f.write(pickle.dumps(model))
        f.close()

        return model

    def get_score(self):
        """this function generates the scroe of the model """
        file = open("score.txt", "r")
        content = file.read()
        file.close()
        return float(content)

    def get_params(self):
        """this function generates params of the model """
        return self.get_score(), type(self.model).__name__, self.model.get_params()
       
    def read_model(self):
        model_file = open("model.txt", "rb")
        content = model_file.read()
        model_file.close()
        return pickle.loads(content)
   
    def predict(self, query):
        return self.model.predict(np.array(query).reshape(1, 7))[0]
   
    def predict_proba(self, query):
        """predict probability"""
        return self.model.predict_proba(np.array(query).reshape(1, 7))[0]

       
class DataManager:
   
    """class containing the data management"""
    def __init__(self):
        self.data = pd.read_csv("data.csv", sep=';')
       
    def inject_false_data(self):
        """we inject data that are losing """
        self.data = self.data[["N1", "N2", "N3", "N4", "N5", "E1", "E2"]].assign(HasWin = np.ones(self.data.shape[0], dtype=int))
       
        shape = 4 * self.data.shape[0]
        false_data_np = np.hstack((np.random.randint(50, size=(shape, 5)), np.random.randint(12, size=(shape, 2)), np.zeros((1, shape), dtype=int).T))
        false_data = pd.DataFrame(
            data=false_data_np,
            columns=["N1", "N2", "N3", "N4", "N5", "E1", "E2", "HasWin"],
            index=list(range(self.data.shape[0], self.data.shape[0] + false_data_np.shape[0]))
        )
        self.data = pd.concat([self.data, false_data])
        print(self.data.head())
        self.data.to_csv("data.csv", sep=';', index=False)
       
    def add(self, data):
        """ we add a line from a csv"""
        data = [int(e) for e in data.split(";")[1:-2]]
        data.append(1)
        data = np.array(data)
        nd = pd.DataFrame(np.array(data).reshape(1,-1), columns=list(self.data))
        self.data = self.data.append(nd)
        self.data.to_csv("data.csv", sep=';', index=False)
        return data
       
 
# iamodel = IAModel()
# iamodel.data_manager.add([1, 2, 3, 4, 5, 6, 7, 0])
# iamodel.predict([1, 2, 3, 4, 5, 6, 8])
# iamodel.predict_proba([1, 2, 3, 4, 5, 6, 8])