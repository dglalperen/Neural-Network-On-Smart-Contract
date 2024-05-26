def integer_sigmoid(x, scale=10000):
    # Constants scaled by 'scale'
    scaled_one = scale

    # More accurate exponential function using integer arithmetic
    # Using the exponential approximation: e^x ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
    # Scale down x to manage overflow in calculations
    x_scaled = -x // scale

    # Calculating the series terms
    x2 = (x_scaled * x_scaled) // scale
    x3 = (x2 * x_scaled) // scale
    x4 = (x3 * x_scaled) // scale

    # Sum the series up to x^4
    scaled_exp = scaled_one + x_scaled + x2 // 2 + x3 // 6 + x4 // 24

    if x < 0:
        scaled_exp = scaled_one * scaled_one // scaled_exp  # since e^-x = 1/e^x

    # Sigmoid function approximation
    result = scaled_one * scaled_one // (scaled_one + scaled_exp)

    return result


# Test the function with a sample input
x_value = 100  # This represents 0.1 in fixed-point
sigmoid_output = integer_sigmoid(x_value)
print("Sigmoid approximation:", sigmoid_output / 10000.0)
