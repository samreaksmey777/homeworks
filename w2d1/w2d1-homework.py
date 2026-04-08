import numpy as np

# 1. Create a function that computes the dot product of two vectors.
def manual_dot(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(a * b)

# 2. Verify the function with two random vectors.

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

manual_dot = manual_dot(a, b)
np_dot = np.dot(a, b)
print(f"Manual dot product of {a} and {b} is: {manual_dot}")
print(f"NumPy dot product of {a} and {b} is: {np_dot}")

print(f"{np_dot}=={manual_dot} is {np_dot == manual_dot}")


