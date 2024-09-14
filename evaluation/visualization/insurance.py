import pandas as pd
import matplotlib.pyplot as plt


def plot_insurance_predictions():
    # Load PyTorch predictions
    pytorch_results = pd.read_csv("../results/insurance_net_results.csv")

    # Smart contract predictions (collected manually)
    smart_contract_results = pd.DataFrame(
        {
            "Inputs": [
                "[3, 41, 1, 62, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "[1, 21, 2, 71, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "[1, 22, 1, 19, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "[3, 56, 1, 50, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "[3, 21, 1, 39, 2, 1, 0, 36, 6, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]",
                "[1, 30, 1, 35, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
            ],
            "Smart Contract Output 1": [
                62432 / 100000,
                14310 / 100000,
                59215 / 100000,
                101052 / 100000,
                2754647 / 100000,
                62405 / 100000,
            ],
            "Smart Contract Output 2": [
                -165121 / 100000,
                -117091 / 100000,
                -96430 / 100000,
                -197276 / 100000,
                -2756470 / 100000,
                -124407 / 100000,
            ],
        }
    )

    # Combine results
    combined_results = pd.DataFrame(
        {
            "Inputs": smart_contract_results["Inputs"],
            "PyTorch Output 1": pytorch_results["Probability Class 0"],
            "PyTorch Output 2": pytorch_results["Probability Class 1"],
            "Smart Contract Output 1": smart_contract_results[
                "Smart Contract Output 1"
            ],
            "Smart Contract Output 2": smart_contract_results[
                "Smart Contract Output 2"
            ],
        }
    )

    # Plot comparison
    combined_results.plot(kind="bar", figsize=(14, 8))
    plt.xlabel("Test Cases")
    plt.ylabel("Values")
    plt.title("InsuranceNet: Comparison of PyTorch and Smart Contract Predictions")
    plt.xticks(range(len(combined_results)), combined_results["Inputs"], rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


plot_insurance_predictions()
