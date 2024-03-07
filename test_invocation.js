const { invokeScript, broadcast } = require("@waves/waves-transactions");
const nodeUrl = "https://nodes-testnet.wavesnodes.com";

// Define your dApp address and the account details
const dAppAddress = "3N3n75UqB8G1GKmXFr4zPhKCjGcqJPRSuJY";
const seed =
  "aisle grit neutral neglect midnight blur energy lady mention gesture engage wheel foster juice domain"; // Be careful with your seed. Don't expose it and don't commit it to your repo.

// Define the function call structure
const call = {
  function: "predict",
  args: [{ type: "integer", value: 1 }], // Change the value to test different inputs
};

// Create and sign the invoke script transaction
const signedTx = invokeScript(
  {
    dApp: dAppAddress,
    call: call,
    chainId: "T", // Use 'W' for Mainnet
    fee: 500000, // Set the appropriate fee
  },
  seed
);

// Broadcast the transaction
broadcast(signedTx, nodeUrl)
  .then((response) => {
    console.log(response);
  })
  .catch((error) => {
    console.error(error);
  });
