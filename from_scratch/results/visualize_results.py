import matplotlib.pyplot as plt


def plot_board(board, predictions, title):
    fig, ax = plt.subplots()
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    plt.title(title)

    for i, (b, p) in enumerate(zip(board, predictions)):
        row, col = divmod(i, 3)
        if b == 1:
            ax.text(col, row, "X", ha="center", va="center", fontsize=20, color="blue")
        elif b == -1:
            ax.text(col, row, "O", ha="center", va="center", fontsize=20, color="red")
        else:
            ax.text(
                col, row, f"{p}", ha="center", va="center", fontsize=12, color="green"
            )

    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.gca().invert_yaxis()
    plt.show()


data = [
    {
        "board": [0, 0, 0, 0, 1, 0, 0, 0, 0],
        "predictions": [7557, 5068, 7505, 7525, 0, 5059, 7531, 7513, 7510],
        "title": "Board 0",
    },
    {
        "board": [-1, -1, 1, 0, -1, -1, 1, 0, 1],
        "predictions": [0, 0, 0, 2518, 0, 11, 0, 7563, 0],
        "title": "Board 1",
    },
    {
        "board": [-1, -1, 1, 0, -1, 0, 1, 0, 0],
        "predictions": [0, 48, 0, 2543, 0, 5057, 0, 7523, 7529],
        "title": "Board 2",
    },
    {
        "board": [-1, -1, 1, 0, 1, 0, 0, 0, 0],
        "predictions": [0, 0, 0, 2563, 0, 2566, 7524, 56, 74],
        "title": "Board 3",
    },
    {
        "board": [-1, -1, 1, 0, 0, 0, 0, 0, 0],
        "predictions": [0, 3, 0, 2529, 5015, 5063, 7502, 2542, 5017],
        "title": "Board 4",
    },
    {
        "board": [-1, 0, 0, 1, 0, 0, 0, 0, -1],
        "predictions": [44, 5045, 2561, 0, 7551, 7510, 5029, 7515, 25],
        "title": "Board 5",
    },
]

for entry in data:
    plot_board(entry["board"], entry["predictions"], entry["title"])
