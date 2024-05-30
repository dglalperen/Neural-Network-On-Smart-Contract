const { invokeScript, broadcast } = require("@waves/waves-transactions");
const readline = require("readline");
const nodeUrl = "https://nodes-testnet.wavesnodes.com";

// Define your dApp address and the account seed
const dAppAddress = "3N3n75UqB8G1GKmXFr4zPhKCjGcqJPRSuJY";
const seed =
  "aisle grit neutral neglect midnight blur energy lady mention gesture engage wheel foster juice domain";

let board = [0, 0, 0, 0, 0, 0, 0, 0, 0];

function printBoard() {
  let boardValue = board.map((b) => (b === -1 ? "X" : b === 1 ? "O" : "-"));
  console.log(boardValue[0] + boardValue[1] + boardValue[2]);
  console.log(boardValue[3] + boardValue[4] + boardValue[5]);
  console.log(boardValue[6] + boardValue[7] + boardValue[8]);
}

function getPossibleMoves() {
  return board.map((b, i) => (b === 0 ? i : null)).filter((i) => i !== null);
}

function checkWon() {
  const winningCombinations = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];

  for (let combination of winningCombinations) {
    const [a, b, c] = combination;
    if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }
  return 0;
}

function determineWinner() {
  const hasWon = checkWon();
  if (hasWon === 1) {
    printBoard();
    console.log("Player has won!");
    process.exit();
  } else if (hasWon === -1) {
    printBoard();
    console.log("AI has won!");
    process.exit();
  }
}

async function predictAI() {
  const callData = {
    function: "predict",
    args: board.map((value) => ({ type: "integer", value })),
  };

  const signedTx = invokeScript(
    {
      dApp: dAppAddress,
      call: callData,
      chainId: "T",
      fee: 500000,
    },
    seed
  );

  try {
    const response = await broadcast(signedTx, nodeUrl);
    // Assuming the response contains the AI's move index
    const moveIndex = response.call.args[0].value; // Modify based on actual response format
    board[moveIndex] = -1;
  } catch (error) {
    console.error("Error:", error);
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function userInput() {
  return new Promise((resolve) => {
    rl.question("Enter number of the field you want to move to: ", (move) => {
      board[parseInt(move) - 1] = 1;
      resolve();
    });
  });
}

(async function testTicTacToe() {
  while (true) {
    printBoard();
    await userInput();
    determineWinner();
    await predictAI();
    determineWinner();
  }
})();
