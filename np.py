import numpy as np

psi_star = np.array([0, 0, 0])
params = np.array([1, 2, 3])
print(np.sum(np.square(params - psi_star)))
print(np.linalg.norm(params - psi_star))

print(np.array(psi_star))