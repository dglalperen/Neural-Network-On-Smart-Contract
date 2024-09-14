import pandas as pd
import matplotlib.pyplot as plt


def plot_xor_predictions(file_name, smart_contract_outputs, model_name):
    # Load PyTorch predictions
    pytorch_results = pd.read_csv(file_name)

    # Smart contract predictions (collected manually)
    smart_contract_results = pd.DataFrame(
        {
            "Inputs": ["[0,0]", "[0,1]", "[1,0]", "[1,1]"],
            "Smart Contract Output": [
                output / 10000 for output in smart_contract_outputs
            ],
        }
    )

    # Combine results
    combined_results = pd.DataFrame(
        {
            "Inputs": smart_contract_results["Inputs"],
            "PyTorch Output": pytorch_results["Output"],
            "Smart Contract Output": smart_contract_results["Smart Contract Output"],
        }
    )

    # Plot comparison
    combined_results.plot(kind="bar", figsize=(12, 6))
    plt.xlabel("Test Cases")
    plt.ylabel("Values")
    plt.title(f"{model_name}: Comparison of PyTorch and Smart Contract Predictions")
    plt.xticks(range(len(combined_results)), combined_results["Inputs"], rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# TwoLayerXORNet
plot_xor_predictions(
    "../results/two_layer_xor_net_results.csv", [6, 7530, 7530, 97], "TwoLayerXORNet"
)
