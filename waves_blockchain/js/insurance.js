const { invokeScript, broadcast } = require("@waves/waves-transactions");
const nodeUrl = "https://nodes-testnet.wavesnodes.com";

// Define your dApp address and the account seed
const dAppAddress = "3N3n75UqB8G1GKmXFr4zPhKCjGcqJPRSuJY";
const seed =
  "aisle grit neutral neglect midnight blur energy lady mention gesture engage wheel foster juice domain";

// Function to create and broadcast a prediction call
function predict(inputs) {
  const callData = {
    function: "predict_insurance",
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
    [
      3.0, 41.0, 1.0, 62.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
      1.0, 21.0, 2.0, 71.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
      1.0, 22.0, 1.0, 19.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
      3.0, 56.0, 1.0, 50.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
      3.0, 21.0, 1.0, 39.0, 2.0, 1.0, 0.0, 36.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
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
