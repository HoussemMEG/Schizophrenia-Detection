import numpy as np

real  = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])

out_1 = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
out_2 = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
out_3 = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1])

voted = out_1 + out_2 + out_3
voted = np.array([1 if x > 1.5 else 0 for x in voted])

print('first output accuracy {:.1f} %'.format(100 - np.logical_xor(out_1, real).mean() * 100))
print('second output accuracy {:.1f} %'.format(100 - np.logical_xor(out_2, real).mean() * 100))
print('third output accuracy {:.1f} %'.format(100 - np.logical_xor(out_3, real).mean() * 100))

print('voted accuracy {:.1f} %'.format(100 - np.logical_xor(voted, real).mean() * 100))
