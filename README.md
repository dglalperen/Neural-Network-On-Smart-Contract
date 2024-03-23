# PyTorch to Waves Smart Contract Converter

This project provides a tool to convert PyTorch neural network models into Waves smart contracts. The generated smart contracts can then be deployed on the Waves blockchain for inference.

## Features

- Convert PyTorch models into Waves smart contracts.
- Support for models with specific layer configurations:
  - Two input neurons in the first layer.
  - Four neurons in the hidden layers.
  - One neuron in the output layer.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/pytorch-to-waves-converter.git
   ```

2. Install the required dependencies:

   ```bash
   pip install torch
   ```

## Usage

1. Place your PyTorch model files (`.pth`) in the `drag_torch_model` directory.

2. Run the `main.py` script to convert the models into Waves smart contracts:

   ```bash
   python main.py
   ```

3. The generated smart contracts will be saved in the same directory with the `.ride` extension.

## Example

Suppose you have a PyTorch model named `my_model.pth`, conforming to the supported layer configuration. After running the conversion script, a Waves smart contract named `my_model.ride` will be generated.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
