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
  console.log("Prediction for [0, 0]");
  try {
    const response = await predict([0, 0]);
    console.log("Response:", response);
  } catch (error) {
    console.error("Error:", error);
  }

  console.log("Prediction for [0, 1]");
  try {
    const response = await predict([0, 1]);
    console.log("Response:", response);
  } catch (error) {
    console.error("Error:", error);
  }

  console.log("Prediction for [1, 0]");
  try {
    const response = await predict([1, 0]);
    console.log("Response:", response);
  } catch (error) {
    console.error("Error:", error);
  }

  console.log("Prediction for [1, 1]");
  try {
    const response = await predict([1, 1]);
    console.log("Response:", response);
  } catch (error) {
    console.error("Error:", error);
  }
}

performPredictions();
