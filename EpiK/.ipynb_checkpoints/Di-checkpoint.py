# features to be added in future:
# 1. priors for hyperparameter in kernel classes
# 2. low interaction models


import torch
import gpytorch
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from scipy.special import binom as binom
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from matplotlib.pyplot import figure


def d(geno1, geno2):
  """build distance tensor between two sets of genotypes
  geno1, geno2: n x L, m x L torch tensors

  """
  geno1_h0 = 1. * (geno1 == 0.)
  geno1_h1 = 1. * (geno1 == 2.)
  geno2_h0 = 1. * (geno2 == 0.)
  geno2_h1 = 1. * (geno2 == 2.)
  S1 = torch.matmul(geno1 % 2, torch.transpose(geno2 % 2, 0, 1))
  S2 = (torch.matmul(geno1_h0, torch.transpose(geno2_h0, 0, 1))
        + torch.matmul(geno1_h1, torch.transpose(geno2_h1, 0, 1)))
  D2 = (torch.matmul(geno1_h0, torch.transpose(geno2_h1, 0, 1))
        + torch.matmul(geno1_h1, torch.transpose(geno2_h0, 0, 1)))
  D1 = L - S1 - S2 - D2

  return torch.stack((S1, S2, D1, D2))


def k(log_lda, log_eta, dvec):
    """
    log_lda, log_eta -- torch tensors
    dvec -- 4 x n x m torch tensor
    """
    lda = torch.exp(log_lda)
    eta = torch.exp(log_eta)
    return (((1 + lda + eta)**(dvec[1] - L / 2))
          * ((1 - lda + eta)**dvec[3])
          * ((1 + eta)**(dvec[0] - L / 2))
          * (1 - eta)**dvec[2])


# import positivity constraint
from gpytorch.constraints import Positive
from gpytorch.constraints import LessThan


class DiKernel(gpytorch.kernels.Kernel):
  """
  Diploid kernel
  lda: epistasis parameter
  eta: dominance parameter
  """
  is_stationary = True

  # We will register the parameter
  def __init__(self,
                lda_constraint=None,
                eta_constraint=None,
                **kwargs):
      super().__init__(**kwargs)

      # register the raw parameter
      self.register_parameter(
          name='raw_lda',
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
      )

      self.register_parameter(
          name='raw_eta',
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
      )

      # set the parameter constraint to be positive
      lda_constraint = LessThan(upper_bound=0.)
      eta_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_lda", lda_constraint)
      self.register_constraint("raw_eta", eta_constraint)

  # set up the 'actual' paramter
  @property
  def lda(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_lda_constraint.transform(self.raw_lda)

  @property
  def eta(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_eta_constraint.transform(self.raw_eta)

  @lda.setter
  def lda(self, value):
      return self._set_lda(value)

  @eta.setter
  def eta(self, value):
      return self._set_eta(value)

  def forward(self, x1, x2, diag=False, **params):
      diff = d(x1, x2)
      K = k(self.lda, self.eta, diff)
      if diag:
        K = K[0]
      return K


class LinKernel(gpytorch.kernels.Kernel):
  """
  Additive kernel
  no free parameters
  calculated using the D tensor
  """

  is_stationary = True

  # We will register the parameter when initializing the kernel
  def __init__(self,
                **kwargs):
      super().__init__(**kwargs)

  def forward(self, x1, x2, **params):
      diff = d(x1, x2)
      return 2 * (diff[1] - diff[-1])


class RBFKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        

        # register the raw parameter
        self.register_parameter(
            name='raw_length', 
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set the parameter constraint to be positive, when nothing is specified
        length_constraint = Positive()
            

        # register the constraint
        self.register_constraint("raw_length", length_constraint)


    # set up actual paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))
        
    def get_diff(self, x1, x2, **params):
        diff = d(x1, x2)
#         diff = (diff[2] + 4*diff[3])
        return diff[0]


    def forward(self, x1, x2, **params):
        diff = d(x1, x2)
        diff = (diff[2] + 4*diff[3])/L
        return torch.exp(-self.length*diff)
      


class DiGPModel(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            DiKernel(), device_ids=range(n_devices),
            output_device=output_device
        )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)    


class LinGPModel(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            LinKernel(), device_ids=range(n_devices),
            output_device=output_device
        )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                RBFKernel(), device_ids=range(n_devices),
                output_device=output_device
            )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



training_iter=200

def train_model(model, likelihood, train_x, train_y, training_iter=training_iter, lr=0.05):
    losses = []
    
    """fitting hyperparameters of model by maximizing marginal log likelihood"""
    # Use the adam optimizer, this includes GaussianLikelihood parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr)


    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
      if i%20==0:
        print('working on iteration {}'.format(i))
      else: pass
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = model(train_x)
      # Calc loss and backprop gradients
      loss = -mll(output, train_y)
      loss.backward()
      losses.append(loss.item())    
      optimizer.step()
      del loss
    return losses
