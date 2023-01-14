import numpy as np
from fitting import fit_spectrum
import matplotlib.pyplot as plt
spectrum = np.loadtxt("../demo_data/spectrum.csv")
ref = np.loadtxt("../demo_data/ref_spectrum.csv")
fitted = fit_spectrum(spectrum, ref, outlier_threshold=0.01)

plt.plot(spectrum, label="spectrum")
plt.plot(ref, label="ref")
plt.plot(fitted, label="fitted")
plt.legend()
plt.show()