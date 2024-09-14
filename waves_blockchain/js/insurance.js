const { invokeScript, broadcast } = require("@waves/waves-transactions");
const nodeUrl = "https://nodes-testnet.wavesnodes.com";

// Define your dApp address and the account seed
const dAppAddress = "3N3n75UqB8G1GKmXFr4zPhKCjGcqJPRSuJY";
const seed =
  "aisle grit neutral neglect midnight blur energy lady mention gesture engage wheel foster juice domain";

// Function to create and broadcast a prediction call
function predict(inputs) {
  const callData = {
    function: "predict",
    args: [
      {
        type: "list",
        value: inputs.map((value) => ({ type: "integer", value })),
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
async function performPredictions() {
  const testInputs = [
    [3, 41, 1, 62, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 21, 2, 71, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 22, 1, 19, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 56, 1, 50, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 21, 1, 39, 2, 1, 0, 36, 6, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
    [1, 30, 1, 35, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 23, 1, 47, 1, 1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 72, 0, 0, 0],
    [1, 24, 1, 35, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0],
    [1, 41, 1, 35, 2, 1, 0, 36, 6, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
    [1, 72, 1, 36, 1, 1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ];

  for (let i = 0; i < testInputs.length; i++) {
    console.log(`Prediction for input ${i + 1}: ${testInputs[i]}`);
    try {
      const response = await predict(testInputs[i]);
      console.log("Response:", response);
    } catch (error) {
      console.error("Error:", error);
    }
  }
}

performPredictions();
