import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(23, 15)
        self.fc2 = nn.Linear(15, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CrashModelEvaluation:

    def __init__(self):
        self.model = Model()
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()

    # Durchläuft das Model mit einer einfachen Vorwärtspropagation
    def predict(self, df):

        tensor = self.prepare_dataframe(df)

        with torch.no_grad():
            predictions = self.model(tensor)
            return predictions

    # Gibt eine Flag zurück die Aussagt, welche Person Schuld hat. 0 = Person_one ist Schuld. 1 = Person_Two
    def processedPrediction(self, df):
        predict = self.predict(df)
        print(predict)
        per_one = predict[0, 0]
        per_two = predict[0, 1]

        if per_one.item() > per_two.item():
            return 0
        else:
            return 1

    # Bringt die Daten auf das richtige Format und wendet den min-max scaler mit den gleichen Parametern an, den es zum lernen hatte
    def prepare_dataframe(self, df):
        scaler = joblib.load("scaler.pkl")
        scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
        numpy_array = scaled_df.to_numpy()
        tensor = torch.tensor(numpy_array, dtype=torch.float32)
        return tensor


# Laden der Daten aus der Excel-Datei
data = pd.read_csv("first_five_X.csv")

# Auswahl einer belibigen Zeile
selected_row = data.iloc[[1]]

# Erstellung der Evaluationsinstanz
evaluator = CrashModelEvaluation()

# Ausführung der Vorhersage
result = evaluator.processedPrediction(selected_row)

# Ergebnis ausgeben
print("Die vorhergesagte Schuld liegt bei Person:", result)
