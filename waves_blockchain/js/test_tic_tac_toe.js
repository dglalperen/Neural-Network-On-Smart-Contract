const { invokeScript, broadcast } = require("@waves/waves-transactions");
const nodeUrl = "https://nodes-testnet.wavesnodes.com";

// Define your dApp address and the account seed
const dAppAddress = "3N3n75UqB8G1GKmXFr4zPhKCjGcqJPRSuJY";
const seed =
  "aisle grit neutral neglect midnight blur energy lady mention gesture engage wheel foster juice domain";

// Function to create and broadcast a move call
function makeMove(board) {
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

  return broadcast(signedTx, nodeUrl);
}

// Example usage
async function performMoves() {
  const boards = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0], // Initial empty board
    [0, 0, 0, 0, 1, 0, 0, 0, 0], // Player O move
    [1, 0, 0, 0, 0, 0, 0, 0, 0], // Player O move
    [1, -1, 0, 0, 0, 0, 0, 0, 0], // AI move
    [1, -1, 0, 1, 0, 0, 0, 0, 0], // Player O move
    [1, -1, -1, 1, 0, 0, 0, 0, 0], // AI move
    [-1, -1, 0, 1, 0, 0, 0, 0, 0], // Player O move (Player is about to win)
    [-1, 0, 0, 1, 0, 0, 0, 0, -1], // AI move (AI blocks player)
    [-1, -1, 1, 0, 0, 0, 0, 0, 0], // Player O move (Player is about to win)
    [-1, -1, 1, 0, 1, 0, 0, 0, 0], // AI move (AI blocks player)
    [-1, -1, 1, 0, -1, 0, 1, 0, 0], // Player O move (Player is about to win)
    [-1, -1, 1, 0, -1, -1, 1, 0, 1], // AI move (AI blocks player)
  ];

  for (let i = 0; i < boards.length; i++) {
    console.log(`Making move with board state: ${boards[i]}`);
    try {
      const response = await makeMove(boards[i]);
      console.log("Response:", response);
    } catch (error) {
      console.error("Error:", error);
    }
  }
}

performMoves();
