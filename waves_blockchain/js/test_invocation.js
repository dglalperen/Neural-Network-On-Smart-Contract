const { invokeScript, broadcast } = require("@waves/waves-transactions");
const nodeUrl = "https://nodes-testnet.wavesnodes.com";

// Define your dApp address and the account details
const dAppAddress = "3N3n75UqB8G1GKmXFr4zPhKCjGcqJPRSuJY";
const dAppAddressIDE = "3NBwmKwBV3F4vBVoAUJfqVhV67oenSHaqRV";
const seed =
  "aisle grit neutral neglect midnight blur energy lady mention gesture engage wheel foster juice domain";

// Define the function call structure
const call00 = {
  function: "predict",
  args: [
    { type: "integer", value: 0 },
    { type: "integer", value: 0 },
  ],
};

const call01 = {
  function: "predict",
  args: [
    { type: "integer", value: 0 },
    { type: "integer", value: 1 },
  ],
};

const call10 = {
  function: "predict",
  args: [
    { type: "integer", value: 1 },
    { type: "integer", value: 0 },
  ],
};

const call11 = {
  function: "predict",
  args: [
    { type: "integer", value: 1 },
    { type: "integer", value: 1 },
  ],
};

// Create and sign the invoke script transaction
const signedTx00 = invokeScript(
  {
    dApp: dAppAddress,
    call: call00,
    chainId: "T",
    fee: 500000,
  },
  seed
);

const signedTx01 = invokeScript(
  {
    dApp: dAppAddress,
    call: call01,
    chainId: "T",
    fee: 500000,
  },
  seed
);

const signedTx10 = invokeScript(
  {
    dApp: dAppAddress,
    call: call10,
    chainId: "T",
    fee: 500000,
  },
  seed
);

const signedTx11 = invokeScript(
  {
    dApp: dAppAddress,
    call: call11,
    chainId: "T",
    fee: 500000,
  },
  seed
);

broadcast(signedTx01, nodeUrl)
  .then((response) => {
    console.log("Response", response);
  })
  .catch((error) => {
    console.error(error);
  });
