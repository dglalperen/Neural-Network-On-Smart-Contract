import sys
import random

from tictacnet import TicTacNet
import torch

board = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

model = TicTacNet()
model.load_state_dict(torch.load('tictactoe.pth'))

def printBoard():
    boardValue = []
    for b in board:
        if b == -1:
            boardValue.append('X')
        elif b == 1:
            boardValue.append('O')
        else:
            boardValue.append('-')

    print(boardValue[0] + boardValue[1] + boardValue[2])
    print(boardValue[3] + boardValue[4] + boardValue[5])
    print(boardValue[6] + boardValue[7] + boardValue[8])

def userInput():
    move = input('enter number of the field you want to move to: ')
    board[int(move) - 1] = 1

def get_valid_move_index_q_value_pairs(q_values, valid_move_indexes):
    valid_q_values = []
    for vmi in valid_move_indexes:
        valid_q_values.append((vmi, q_values[vmi].item()))

    return valid_q_values

def aiMove():
    tensor = torch.tensor(board, dtype=torch.float)
    results = model(tensor)
    possibleMoves = getPossibleMoves()
    valid_q_values = get_valid_move_index_q_value_pairs(results, possibleMoves)
    move, value = max(valid_q_values, key=lambda pair: pair[1])
    if random.random() < 0.1:
        valid_q_values.remove((move, value))
        move, value = max(valid_q_values, key=lambda pair: pair[1])
    board[move] = -1

def getPossibleMoves():
    possibleMoves = []
    for i in range(0, len(board)):
        if board[i] == 0:
            possibleMoves.append(i)

    if len(possibleMoves) == 0:
        printBoard()
        print('Draw!')
        sys.exit()

    return possibleMoves

def checkWon():
    if board[0] == board[1] == board[2]:
        return board[0]
    elif board[3] == board[4] == board[5]:
        return board[3]
    elif board[6] == board[7] == board[8]:
        return board[6]
    elif board[0] == board[3] == board[6]:
        return board[0]
    elif board[1] == board[4] == board[7]:
        return board[1]
    elif board[2] == board[5] == board[8]:
        return board[2]
    elif board[0] == board[4] == board[8]:
        return board[0]
    elif board[2] == board[4] == board[6]:
        return board[2]
    else:
        return 0

def determineWinner():
    hasWon = checkWon()
    if hasWon == 1:
        printBoard()
        print('Player has won!')
        sys.exit()
    elif hasWon == -1:
        printBoard()
        print('AI has won!')
        sys.exit()

while True:
    printBoard()
    userInput()
    determineWinner()
    aiMove()
    determineWinner()
