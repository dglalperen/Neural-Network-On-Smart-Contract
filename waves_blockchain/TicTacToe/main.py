import sys
import random

from tictacnet import TicTacNet
import torch

board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

model = TicTacNet()
model.load_state_dict(torch.load("tictactoe.pth"))


def printBoard():
    boardValue = []
    for b in board:
        if b == -1:
            boardValue.append("X")
        elif b == 1:
            boardValue.append("O")
        else:
            boardValue.append("-")

    print(boardValue[0] + boardValue[1] + boardValue[2])
    print(boardValue[3] + boardValue[4] + boardValue[5])
    print(boardValue[6] + boardValue[7] + boardValue[8])


def userInput():
    while True:
        move = input("Enter the number of the field you want to move to (1-9): ")
        try:
            move = int(move) - 1
            if move < 0 or move > 8:
                print("Invalid input. Please enter a number between 1 and 9.")
            elif board[move] != 0:
                print("This position is already taken. Please choose another one.")
            else:
                board[move] = 1
                break
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 9.")


def get_valid_move_index_q_value_pairs(q_values, valid_move_indexes):
    valid_q_values = []
    for vmi in valid_move_indexes:
        valid_q_values.append((vmi, q_values[vmi].item()))
    return valid_q_values


def aiMove():
    tensor = torch.tensor(board, dtype=torch.float)
    results = model(tensor)
    print(results)
    possibleMoves = getPossibleMoves()
    valid_q_values = get_valid_move_index_q_value_pairs(results, possibleMoves)
    move, value = max(valid_q_values, key=lambda pair: pair[1])
    if random.random() < 0.1:  # Adding randomness for exploration
        valid_q_values.remove((move, value))
        move, value = max(valid_q_values, key=lambda pair: pair[1])
    board[move] = -1


def getPossibleMoves():
    possibleMoves = []
    for i in range(len(board)):
        if board[i] == 0:
            possibleMoves.append(i)

    if len(possibleMoves) == 0:
        printBoard()
        print("Draw!")
        sys.exit()
    return possibleMoves


def checkWon():
    win_conditions = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    for a, b, c in win_conditions:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    return 0


def determineWinner():
    hasWon = checkWon()
    if hasWon == 1:
        printBoard()
        print("Player has won!")
        sys.exit()
    elif hasWon == -1:
        printBoard()
        print("AI has won!")
        sys.exit()


while True:
    printBoard()
    userInput()
    determineWinner()
    aiMove()
    determineWinner()
