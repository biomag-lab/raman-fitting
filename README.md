# raman-fitting

Uses `skimage`, please install it with pip:
```
pip install scikit-image
```

If you want to run the demo, you will also need `matplotlib`:
```
pip install matplotlib
```

## Usage
```
from fitting import fit_spectrum
fit_spectrum(spectrum, ref)
```
where `spectrum` is the spectrum we would like to transform, and `ref` is the reference spectrum (we want `spectrum` to be "similar" to `ref`)

Optionally you can set the polynomial order with `order` (the default is `5`), and the outlier threshould with `outlier_threshold` (default value is `1.0`). The latter controls the threshold for the difference between the reference and the spectrum, above which it is considered as a significant peak (thus it will be ignored on fitting).

```
from fitting import fit_spectrum
fit_spectrum(spectrum, ref, order=polyorder, outlier_threshold=threshold)
```

To try the demo in the `src` folder, simply run
```
python demo.py
```