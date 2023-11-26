# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 09:01:29 2023

@author: Max
"""

import numpy as np
from scipy import linalg
from scipy.stats import gaussian_kde

class GaussianKdeWithEpsilon(gaussian_kde):
    '''
    def __init__(self, dataset, bw_method=None):
        super().__init__(dataset, bw_method)

    def _compute_covariance(self):
        super()._compute_covariance()
        min_eig = np.min(np.real(linalg.eigvals(self.covariance)))
        if min_eig < 0:
            self.covariance -= min_eig * np.eye(*self.covariance.shape)
    '''
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-10  # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            self._data_covariance += self.EPSILON * np.eye(self._data_covariance.shape[0])
            self._data_cho_cov = linalg.cholesky(self._data_covariance,
                                                 lower=True)

        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov
                                        * np.sqrt(2*np.pi))).sum()