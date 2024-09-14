import pandas as pd
import matplotlib.pyplot as plt


def plot_tictac_predictions():
    # Load PyTorch predictions
    pytorch_results = pd.read_csv("../results/tic_tac_toe_net_results.csv")

    # Smart contract predictions (collected manually)
    smart_contract_results = pd.DataFrame(
        {
            "Inputs": [
                "[0,0,0,0,1,0,0,0,0]",
                "[-1,-1,1,0,-1,-1,1,0,1]",
                "[-1, -1,1,0, -1,0,1,0,0]",
                "[-1, -1,1,0,1,0,0,0,0]",
                "[-1, -1,1,0,0,0,0,0,0]",
                "[-1,0,0,1,0,0,0,0, -1]",
            ],
            "Smart Contract Outputs": [
                [
                    8880 / 10000,
                    7554 / 10000,
                    7592 / 10000,
                    8751 / 10000,
                    200 / 10000,
                    7529 / 10000,
                    8776 / 10000,
                    7614 / 10000,
                    7605 / 10000,
                ],
                [0, 20 / 10000, 0, 131 / 10000, 0, 44 / 10000, 0, 8905 / 10000, 0],
                [
                    0,
                    192 / 10000,
                    0,
                    36 / 10000,
                    0,
                    7522 / 10000,
                    0,
                    7646 / 10000,
                    8767 / 10000,
                ],
                [
                    0,
                    96 / 10000,
                    0,
                    76 / 10000,
                    0,
                    83 / 10000,
                    7648 / 10000,
                    18 / 10000,
                    72 / 10000,
                ],
                [
                    0,
                    15 / 10000,
                    0,
                    8 / 10000,
                    5030 / 10000,
                    7540 / 10000,
                    7582 / 10000,
                    34 / 10000,
                    5035 / 10000,
                ],
                [
                    179 / 10000,
                    5090 / 10000,
                    72 / 10000,
                    0,
                    8856 / 10000,
                    7605 / 10000,
                    5059 / 10000,
                    7621 / 10000,
                    100 / 10000,
                ],
            ],
        }
    )

    # Aggregate smart contract results for comparison
    smart_contract_aggregated = pd.DataFrame(
        {"Inputs": smart_contract_results["Inputs"]}
    )
    for i in range(9):
        smart_contract_aggregated[f"Smart Contract Output {i+1}"] = [
            output[i] for output in smart_contract_results["Smart Contract Outputs"]
        ]

    # Aggregate PyTorch results for comparison
    pytorch_aggregated = pd.DataFrame({"Inputs": pytorch_results.iloc[:, 0]})
    for i in range(9):
        pytorch_aggregated[f"PyTorch Output {i+1}"] = pytorch_results.iloc[:, i + 1]

    # Plot comparison
    for i in range(9):
        plt.figure(figsize=(12, 8))

        plt.bar(
            ["Smart Contract"] * len(smart_contract_aggregated),
            smart_contract_aggregated[f"Smart Contract Output {i+1}"],
            color="blue",
            alpha=0.6,
            label="Smart Contract",
        )

        plt.bar(
            ["PyTorch"] * len(pytorch_aggregated),
            pytorch_aggregated[f"PyTorch Output {i+1}"],
            color="orange",
            alpha=0.6,
            label="PyTorch",
        )

        plt.xlabel("Output Neurons")
        plt.ylabel("Value")
        plt.title(f"TicTacToeNet: Comparison for Output Neuron {i+1}")
        plt.legend(loc="best")
        plt.show()


plot_tictac_predictions()
