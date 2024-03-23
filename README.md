# PyTorch to Waves Smart Contract Converter

This Python script converts PyTorch neural network models into smart contracts compatible with the Waves blockchain platform. The generated smart contracts can then be deployed on the Waves platform to make predictions based on the trained neural network models.

## Features

- Converts PyTorch neural network models into Waves-compatible smart contracts.
- Handles both two-layer and three-layer XOR neural network models.
- Automatically generates smart contract code based on the structure and parameters of the input PyTorch models.
- Supports sigmoid activation function and fractional arithmetic for compatibility with the Waves Ride language.

## Requirements

- Python 3.x
- PyTorch
- Numpy

## Usage

1. Install the required Python packages:

```bash
pip install torch numpy
```

2. Define and train your PyTorch neural network models. Ensure that the models are compatible with the provided converter script.

3. Place your trained PyTorch model files (`.pth`) in the appropriate directories (`./TwoLayerXOR/` and `./ThreeLayerXOR/`).

4. Run the converter script:

```bash
python pytorch_to_waves_contract.py
```

5. The converted smart contract code will be printed to the console. Copy the code and deploy it to the Waves platform.

## File Structure

- `pytorch_to_waves_contract.py`: Main Python script for converting PyTorch models to Waves smart contracts.
- `TwoLayerXOR/`: Directory containing trained PyTorch models and scripts for the two-layer XOR neural network.
- `ThreeLayerXOR/`: Directory containing trained PyTorch models and scripts for the three-layer XOR neural network.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project was inspired by the need to deploy machine learning models on blockchain platforms.
- The Waves platform documentation and community provided valuable insights into creating compatible smart contracts.
