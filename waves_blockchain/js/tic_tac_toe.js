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
    args: [
      {
        type: "list",
        value: board.map((value) => ({ type: "integer", value })),
      },
    ],
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
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [-1, -1, 1, 0, -1, -1, 1, 0, 1],
    [-1, -1, 1, 0, -1, 0, 1, 0, 0],
    [-1, -1, 1, 0, 1, 0, 0, 0, 0],
    [-1, -1, 1, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 1, 0, 0, 0, 0, -1],
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
