import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import torch.nn as nn
import torch.nn.functional as F


class InsuranceNet(nn.Module):
    def __init__(self):
        super(InsuranceNet, self).__init__()
        self.fc1 = nn.Linear(23, 15)
        self.fc2 = nn.Linear(15, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CrashModelEvaluation:
    def __init__(self):
        self.model = InsuranceNet()
        self.model.load_state_dict(torch.load("insurance_net.pth"))
        self.model.eval()

    def predict(self, df):
        tensor = self.prepare_dataframe(df)
        with torch.no_grad():
            z1 = self.model.fc1(tensor)
            print("Intermediate output after fc1:", z1)
            a1 = F.relu(z1)
            print("Intermediate output after ReLU:", a1)
            z2 = self.model.fc2(a1)
            print("Final output before activation:", z2)
            return z2

    def processedPrediction(self, df):
        predict = self.predict(df)
        per_one = predict[0, 0]
        per_two = predict[0, 1]
        if per_one.item() > per_two.item():
            return 0
        else:
            return 1

    def prepare_dataframe(self, df):
        scaler = joblib.load("scaler.pkl")
        scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
        numpy_array = scaled_df.to_numpy()
        tensor = torch.tensor(numpy_array, dtype=torch.float32)
        return tensor


# Test data (Replace these inputs with the ones you want to test)
test_inputs = [
    [
        3.0,
        41.0,
        1.0,
        62.0,
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.0,
        21.0,
        2.0,
        71.0,
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.0,
        22.0,
        1.0,
        19.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.0,
        56.0,
        1.0,
        50.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        3.0,
        21.0,
        1.0,
        39.0,
        2.0,
        1.0,
        0.0,
        36.0,
        6.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
]

# Convert test inputs to DataFrame
columns = [
    "LGT_COND",
    "PER_ONE_AGE",
    "PER_ONE_SEX",
    "PER_TWO_AGE",
    "PER_TWO_SEX",
    "VEH_ONE_IMP",
    "VEH_TWO_IMP",
    "VEH_ONE_DR_SF1",
    "VEH_ONE_DR_SF2",
    "VEH_ONE_DR_SF3",
    "VEH_ONE_DR_SF4",
    "VEH_TWO_DR_SF1",
    "VEH_TWO_DR_SF2",
    "VEH_TWO_DR_SF3",
    "VEH_TWO_DR_SF4",
    "VEH_ONE_DR_VIO1",
    "VEH_ONE_DR_VIO2",
    "VEH_ONE_DR_VIO3",
    "VEH_ONE_DR_VIO4",
    "VEH_TWO_DR_VIO1",
    "VEH_TWO_DR_VIO2",
    "VEH_TWO_DR_VIO3",
    "VEH_TWO_DR_VIO4",
]

data = pd.DataFrame(test_inputs, columns=columns)

# Create evaluation instance
evaluator = CrashModelEvaluation()

# Perform predictions
for index, row in data.iterrows():
    prediction = evaluator.processedPrediction(row.to_frame().T)
    print(f"Prediction for input {index + 1}: {prediction}")
