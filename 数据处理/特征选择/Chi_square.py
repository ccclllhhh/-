import numpy as np
from scipy.stats import chi2_contingency

observed_values = np.array([[10, 15, 20], [5, 10, 15]])

chi2, p, dof, expected = chi2_contingency(observed_values)

print("Chi-square value:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)