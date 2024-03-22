// SPDX-License-Identifier: UNKNOWN 
pragma solidity ^0.8.0;

contract XOR_Predictions {
    mapping(bytes32 => bool) public predictions;

    function storePrediction(bytes32 inputHash, bool prediction) public {
        predictions[inputHash] = prediction;
    }

    function verifyPrediction(bytes32 inputHash, bool prediction) public view returns (bool) {
        return predictions[inputHash] == prediction;
    }
}