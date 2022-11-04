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


def set_params(geno):
    global N, L, D_t, K1
    N, L = geno.shape
    D_t = d(geno.cuda(), geno.cuda()).double()
    K1 = K(1,0,D_t)
    


def d(geno1, geno2):
  """build distance tensor between two sets of genotypes
  geno1, geno2: n x L, m x L torch tensors
  
  """
  geno1_h0 = 1.*(geno1 == 0.)
  geno1_h1 = 1.*(geno1 == 2.)
  geno2_h0 = 1.*(geno2 == 0.)
  geno2_h1 = 1.*(geno2 == 2.)
  S1 = torch.matmul(geno1%2, torch.transpose(geno2%2, 0, 1))
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
    return (((1 + lda + eta)**(dvec[1] - L/2))
          *((1 - lda + eta)**dvec[3])
          *((1 + eta)**(dvec[0] - L/2)) 
          * (1-eta)**dvec[2])


def d_idx(x1, x2):
    "return D matrix from precomputed D_t, x_1, x_2: indices"
#     print(x1.shape, x2.shape)
    x1 = torch.squeeze(x1)
    x2 = torch.squeeze(x2)    
    return torch.index_select(torch.index_select(D_t, 1, x1), 2, x2)


    
# import positivity constraint
from gpytorch.constraints import Positive
from gpytorch.constraints import LessThan


class DiKernel(gpytorch.kernels.Kernel):
  """Diploid kernel"""

  is_stationary = True

  # We will register the parameter when initializing the kernel
  def __init__(self, 
                lda_prior=None, lda_constraint=None, 
                eta_prior=None, eta_constraint=None,
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

      # set the parameter constraint to be positive, when nothing is specified
      if lda_constraint is None:
          lda_constraint = LessThan(upper_bound=0.)

      if eta_constraint is None:
          eta_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_lda", lda_constraint)
      self.register_constraint("raw_eta", eta_constraint)

      
  # now set up the 'actual' paramter
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
      
      

    
    
class DiKernel2(gpytorch.kernels.Kernel):
  """Diploid kernel"""
  is_stationary = True
  #Register the parameter when initializing the kernel
  def __init__(self, 
                lda_prior=None, lda_constraint=None, 
                eta_prior=None, eta_constraint=None,
                ld1_prior=None, ld1_constraint=None, 
                et1_prior=None, et1_constraint=None,               
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

      self.register_parameter(
          name='raw_ld1', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
      )

      self.register_parameter(
          name='raw_et1', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
      )

    # set the parameter constraint to be positive, when nothing is specified
      if lda_constraint is None:
          lda_constraint = LessThan(upper_bound=0.)

      if eta_constraint is None:
          eta_constraint = LessThan(upper_bound=0.)

      if ld1_constraint is None:
          ld1_constraint = LessThan(upper_bound=0.)

      if et1_constraint is None:
          et1_constraint = LessThan(upper_bound=0.)
            
      # register the constraint
      self.register_constraint("raw_lda", lda_constraint)
      self.register_constraint("raw_eta", eta_constraint)
      self.register_constraint("raw_ld1", ld1_constraint)
      self.register_constraint("raw_et1", et1_constraint)

      
  # now set up the 'actual' paramter
  @property
  def lda(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_lda_constraint.transform(self.raw_lda)

  @property
  def eta(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_eta_constraint.transform(self.raw_eta)

  @property
  def ld1(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_ld1_constraint.transform(self.raw_ld1)

  @property
  def et1(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_et1_constraint.transform(self.raw_et1)

  @lda.setter
  def lda(self, value):
      return self._set_lda(value)

  @eta.setter
  def eta(self, value):
      return self._set_eta(value)

  @ld1.setter
  def ld1(self, value):
      return self._set_ld1(value)

  @eta.setter
  def et1(self, value):
      return self._set_et1(value)


  def forward(self, x1, x2, **params):
      diff = d(x1, x2)
      return k(self.lda, self.eta, diff) + k(self.ld1, self.et1, diff)



class DiGPModel2(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = DiKernel2()

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class LinKernel(gpytorch.kernels.Kernel):
  """Additive kernel"""

  is_stationary = True

  # We will register the parameter when initializing the kernel
  def __init__(self, 
                **kwargs):
      super().__init__(**kwargs)

  def forward(self, x1, x2, **params):
      diff = d(x1, x2)
      return 2*(diff[1] - diff[-1])


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


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class RBFKernel(gpytorch.kernels.Kernel):
	"""
	RBF kernel
	length: hyperparameter
	"""
    
    is_stationary = True
    

    def __init__(self, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_length", length_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )

    #Set up the 'actual' paramter
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

    def forward(self, x1, x2, **params):
        diff = d(x1, x2)
        diff = (diff[2] + 4*diff[3])/L
        return torch.exp(-self.length*diff)
      

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
        print(i)
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



def train_model_cv(model, likelihood, train_x, train_y, val_x, val_y, training_iter=training_iter, lr=0.05):
    losses = []

    """fitting hyperparameters of model by maximizing marginal log likelihood"""
    # Use the adam optimizer, this includes GaussianLikelihood parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    for i in range(training_iter):
        if i%10 == 0:
            print("working on iteration %f"%i)
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        model.eval()
        f_preds = model(val_x).mean

        # Calc loss and backprop gradients
        loss = torch.norm(f_preds - val_y)
        model.train()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        del loss

    return losses



  #######Plotting functions

def autocorr_h0(d2, log_lda, log_eta):
  """calculate distance correlation when only S2 and D2 are nonzero, 
  i.e. homozygous distance"""
  return (((1 - 2*np.exp(log_lda) + np.exp(log_eta))
          /(1 + 2*np.exp(log_lda) + np.exp(log_eta)))**d2 
          - 1/(1 + 2*np.exp(log_lda) + np.exp(log_eta))**L)

def plot_vc(log_lda, log_eta):
  """matrix plot of variance components of different orders
  log_lda -- torch tensor
  log_eta -- torch tensor
  """
  vars = [[np.exp(log_lda)**n * np.exp(log_eta)**m for n in range(L)] 
        for m in range(L)]
  vars = np.array(vars)
  #get rid of low skew traingular part
  vars = vars[:,::-1]
  vars[np.tril_indices(L, k=-1)] = "nan"
  vars = vars[:,::-1]
  vars[0,0] = 'nan'

  #variance components
  omegak = vars*mk
  display(pd.DataFrame(np.round(vars[:10, :10], decimals=3)))

  plt.rcParams["figure.figsize"] = [5,5]
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(omegak[:10, :10])
  fig.colorbar(cax)

  plt.show()


def binom(n,k):
    return torch.exp(torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1))

def K(na, nd, dvec):
    k_calc = torch.zeros(dvec.shape[1:]).cuda()
    na = torch.tensor(na).cuda()
    nd = torch.tensor(nd).cuda()

    for q in range(na+1):
        for p in range(nd+1):
            p_ = torch.tensor(p).cuda()
            q_ = torch.tensor(q).cuda()
            k_calc += (-1)**(p+q)*binom(dvec[3], q_)*binom(dvec[1], na-q_)*binom(dvec[2], p_)*binom(L - dvec[2] - na, nd-p_)
            
    return 2**na * k_calc

def K1(dvec):
    return 2*(dvec[1] - dvec[-1])


def make_elem_Ks(max_order):
    global K_t
    K_list = {}
    for k in range(max_order+1):
        for j in range(max_order - k + 1):
            K_list[j,k] = K(j,k,D_t)

    K_t = torch.stack(list(K_list.values())[1:], dim=2)
    K_t = K_t.double()

    
def K_t_idx(x1, x2):
    x1 = torch.squeeze(x1)
    x2 = torch.squeeze(x2)    
    return torch.index_select(torch.index_select(K_t, 0, x1), 1, x2)

def K_1_idx(x1, x2):
    x1 = torch.squeeze(x1)
    x2 = torch.squeeze(x2)    
    return torch.index_select(torch.index_select(K1, 0, x1), 1, x2)


from gpytorch.constraints import Positive
from gpytorch.constraints import LessThan


class DiKernel_l(gpytorch.kernels.Kernel):
  """Diploid kernel"""
  is_stationary = True
  # Register the parameter when initializing the kernel

  def __init__(self,
               sigma_prior=None, sigma_constraint=None,
            **kwargs):
      super().__init__(**kwargs)

      self.register_parameter(
          name='raw_sigma',
          parameter=torch.nn.Parameter(torch.zeros(K_t.shape[-1]).double().cuda())
      )
                             
      # set the parameter constraint to be positive, when nothing is specified
      if sigma_constraint is None:
          sigma_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_sigma", sigma_constraint)

  # now set up the 'actual' paramter
  @property
  def sigma(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_sigma_constraint.transform(self.raw_sigma)
  

  @sigma.setter
  def sigma(self, value):
      return self._set_sigma(value)

  def forward(self, x1, x2, **params):
      K_t_sub = K_t_idx(x1, x2)
#       return K_t_sub
      return torch.matmul(K_t_sub, torch.exp(self.sigma))



class DiKernel_mix(gpytorch.kernels.Kernel):
  """Diploid kernel"""
  is_stationary = True
  # Register the parameter when initializing the kernel

  def __init__(self,
            lda_prior=None, lda_constraint=None, 
            eta_prior=None, eta_constraint=None,               
            sigma_prior=None, sigma_constraint=None,
            **kwargs):
      super().__init__(**kwargs)

      self.register_parameter(
          name='raw_sigma',
          parameter=torch.nn.Parameter(torch.zeros(K_t.shape[-1]).double().cuda())
      )
                             
      # set the parameter constraint to be positive, when nothing is specified
      if sigma_constraint is None:
          sigma_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_sigma", sigma_constraint)
        
      ### Lda and Eta
      # register the raw parameter
      self.register_parameter(
          name='raw_lda', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
      )

      self.register_parameter(
          name='raw_eta', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
      )

      # set the parameter constraint to be positive, when nothing is specified
      if lda_constraint is None:
          lda_constraint = LessThan(upper_bound=0.)

      if eta_constraint is None:
          eta_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_lda", lda_constraint)
      self.register_constraint("raw_eta", eta_constraint)

        
  # now set up the 'actual' paramter
  @property
  def sigma(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_sigma_constraint.transform(self.raw_sigma)
  

  @sigma.setter
  def sigma(self, value):
      return self._set_sigma(value)

  @property
  def lda(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_lda_constraint.transform(self.raw_lda).cuda()

  @property
  def eta(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_eta_constraint.transform(self.raw_eta).cuda()

  @lda.setter
  def lda(self, value):
      return self._set_lda(value)

  @eta.setter
  def eta(self, value):
      return self._set_eta(value)


  def forward(self, x1, x2, **params):
      K_t_sub = K_t_idx(x1, x2)
      diff = d_idx(x1, x2)
#       return K_t_sub
      return torch.matmul(K_t_sub, torch.exp(self.sigma)) + k(self.lda, self.eta, diff)
        
    
    
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

class DiGPModel_l(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = DiKernel_l()

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DiGPModel_mix(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = DiKernel_mix()

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)