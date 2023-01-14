import numpy as np
from skimage.transform import EuclideanTransform
from scipy.optimize import least_squares


class CustomPolyTransform(EuclideanTransform):
    def __init__(self, matrix=None, order=5, loss_fn=None, outlier_threshold=1.0):
        if order is not None and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
            self.order = matrix.shape[0]-2
        else:
            # default to an identity transform
            self.order = order if order is not None else 2
            self.params = np.eye(self.order+2, self.order+2)
        self.loss_fn = loss_fn
        self.outlier_threshold = outlier_threshold


    def estimate(self, src, dst):
        A = np.ones((len(src), self.order+2))
        A[:, 0] = src[:, 0]
        for i in range(self.order+1):
            A[:, i+1] = src[:, 1]**(self.order-i)
        B = dst[:, 0]
        # param_row = np.linalg.lstsq(A, B, rcond=None)[0]
        # self.params[0, :] = param_row
        res = least_squares(CustomPolyTransform.opt_func(A, B), np.eye(self.order+2, 1).reshape(-1), loss="huber", f_scale=self.outlier_threshold)
        self.params[0, :] = res.x
        return True

    def _apply_mat(self, coords, matrix):
        # if self.hack:
        #     coords = coords[:, coords.shape[1]//2]
        coords = np.array(coords, copy=False, ndmin=2)
        x, y = coords.transpose()
        src = np.vstack((x,) + tuple(y**(self.order-i) for i in range(self.order+1)))
        dst = src.T @ matrix.T
        dst[dst[:, -1] == 0, -1] = np.finfo(float).eps
        dst[:, :-1] /= dst[:, -1:]

        return dst[:, :-1:dst.shape[-1]-1]

    def residuals(self, src, dst):
        if self.loss_fn is None:
            return np.sqrt(np.sum((self(src) - dst)**2, axis=1))
        return 1/2 * np.sum(self.loss_fn(dst, self(src)), axis=1)

    @staticmethod
    def opt_func(src, dest):
        def opt_func(params):
            return (src@params)-dest
        return opt_func


def fit_spectrum(x, ref, order=5, outlier_threshold=1.0) -> np.ndarray:
    ref_extended = np.zeros((len(x), 2))
    ref_extended[:, 0] = ref
    ref_extended[:, 1] = np.linspace(0, 1, len(x))
    x_extended = np.concatenate([x.reshape(-1, 1), np.arange(1600).reshape(-1, 1) / 1600], 1)
    tform = CustomPolyTransform(order=order, outlier_threshold=outlier_threshold)
    tform.estimate(x_extended, ref_extended)
    return tform(x_extended)[:, 0]
