import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class Pred_Pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artificats', 'model.pkl')
            processor_path = os.path.join('artificats', 'processor.pkl')

            model = load_object(file_path=model_path)
            transformer = load_object(file_path=processor_path)

            data_trans = transformer.transform(features)
            output = model.predict(data_trans)
            return output
        except Exception as e:
            raise CustomException(e, sys)
        
class input_data:
    def __init__(self,
                 Age: int,
                 Income: int,
                 Emp_length: float,
                 Amount: int,
                 Rate: float,
                 Status: int,
                 Percent_income: float,
                 Cred_length: int
    ):
        self.Age = Age,
        self.Income = Income,
        self.Emp_length = Emp_length,
        self.Amount = Amount,
        self.Rate = Rate,
        self.Status = Status,
        self.Percent_income = Percent_income,
        self.Cred_length = Cred_length
        

    def transfrom_data_as_dataframe(self):
        try:
            user_input_data_dict= {
                "Age": [self.Age],
                "Income": [self.Income],
                "Emp_length": [self.Emp_length],
                "Amount": [self.Amount],
                "Rate": [self.Rate],
                "Status": [self.Status],
                "Percent_income": [self.Percent_income],
                "Cred_length": [self.Cred_length]
            }
            return pd.DataFrame(user_input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)


                 
    
