import torch
import pandas as pd
import joblib
import torch.nn as nn
import torch.nn.functional as F
import os


class InsuranceNet(nn.Module):
    def __init__(self):
        super(InsuranceNet, self).__init__()
        self.fc1 = nn.Linear(23, 15)
        self.fc2 = nn.Linear(15, 2)
        self.activations = {"fc1": "relu", "fc2": None}

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def prepare_dataframe(df):
    scaler = joblib.load("pth/scaler.pkl")
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    tensor = torch.tensor(scaled_df.to_numpy(), dtype=torch.float32)
    return tensor


def evaluate_model(model, df):
    tensor = prepare_dataframe(df)
    with torch.no_grad():
        outputs = model(tensor)
    return outputs


def save_results(df, outputs, file_name="insurance_net_results.csv"):
    predicted_classes = torch.argmax(outputs, dim=1).numpy()
    predicted_probabilities = F.softmax(outputs, dim=1).numpy()
    df["Predicted Class"] = predicted_classes
    df["Probability Class 0"] = predicted_probabilities[:, 0]
    df["Probability Class 1"] = predicted_probabilities[:, 1]
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", file_name), index=False)


if __name__ == "__main__":
    # Load the model
    model = InsuranceNet()
    model.load_state_dict(torch.load("pth/insurance_net.pth"))
    model.eval()

    # Load test data
    data = [
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
        [
            1.0,
            30.0,
            1.0,
            35.0,
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
            3.0,
            23.0,
            1.0,
            47.0,
            1.0,
            1.0,
            0.0,
            6.0,
            0.0,
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
            72.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.0,
            24.0,
            1.0,
            35.0,
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
            58.0,
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
            41.0,
            1.0,
            35.0,
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
        [
            1.0,
            72.0,
            1.0,
            36.0,
            1.0,
            1.0,
            0.0,
            6.0,
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
    ]
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
    test_data = pd.DataFrame(data, columns=columns)

    # Evaluate model
    outputs = evaluate_model(model, test_data)

    # Save results
    save_results(test_data, outputs)
