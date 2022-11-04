#!/usr/bin/env python3

import warnings
import os
from abc import abstractmethod
from copy import deepcopy

import torch
from torch.nn import ModuleList

import gpytorch
from gpytorch import settings
from gpytorch.constraints import Positive
from gpytorch.lazy import LazyEvaluatedKernelTensor, ZeroLazyTensor, delazify, lazify
from gpytorch.models import exact_prediction_strategies
from gpytorch.module import Module
from gpytorch.utils.broadcasting import _mul_broadcast_shape


def default_postprocess_script(x):
    return x


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



class Distance1(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
#         adjustment = x1.mean(-2, keepdim=True)
#         x1 = x1 - adjustment
#         x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x1_pad = torch.ones_like(x1_norm)
#         if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
#             x2_norm, x2_pad = x1_norm, x1_pad
#         else:
#             x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#             x2_pad = torch.ones_like(x2_norm)
#         x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
#         x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
#         res = x1_.matmul(x2_.transpose(-2, -1))
        res = x1.matmul(x2.transpose(-2, -1))        

        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            pass
#             res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
        res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if postprocess else res

    

    
class Kernel(Module):

    has_lengthscale = False

    def __init__(
        self,
        ard_num_dims=None,
        batch_shape=torch.Size([]),
        active_dims=None,
        lengthscale_prior=None,
        lengthscale_constraint=None,
        eps=1e-6,
        **kwargs,
    ):
        super(Kernel, self).__init__()
        self._batch_shape = batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.eps = eps

        param_transform = kwargs.get("param_transform")

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformation, specify a different 'lengthscale_constraint' instead.",
                DeprecationWarning,
            )

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is not None:
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )

            self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    @abstractmethod
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        raise NotImplementedError()

    @property
    def batch_shape(self):
        kernels = list(self.sub_kernels())
        if len(kernels):
            return _mul_broadcast_shape(self._batch_shape, *[k.batch_shape for k in kernels])
        else:
            return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, val):
        self._batch_shape = val

    @property
    def dtype(self):
        if self.has_lengthscale:
            return self.lengthscale.dtype
        else:
            for param in self.parameters():
                return param.dtype
            return torch.get_default_dtype()

    @property
    def is_stationary(self) -> bool:
        """
        Property to indicate whether kernel is stationary or not.
        """
        return self.has_lengthscale

    @property
    def lengthscale(self):
        if self.has_lengthscale:
            return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        else:
            return None

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value):
        if not self.has_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")

        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)

        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def local_load_samples(self, samples_dict, memo, prefix):
        num_samples = next(iter(samples_dict.values())).size(0)
        self.batch_shape = torch.Size([num_samples]) + self.batch_shape
        super().local_load_samples(samples_dict, memo, prefix)

    def covar_dist(
        self,
        x1,
        x2,
        diag=False,
        last_dim_is_batch=False,
        square_dist=True,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params,
    ):

        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if not self.distance_module or self.distance_module._postprocess != dist_postprocess_func:
            self.distance_module = Distance1(dist_postprocess_func)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                if postprocess:
                    res = dist_postprocess_func(res)
                return res
            else:
                res = torch.norm(x1 - x2, p=2, dim=-1)
                if square_dist:
                    res = res.pow(2)
            if postprocess:
                res = dist_postprocess_func(res)
            return res

        elif square_dist:
            res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
        else:
            res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2)

        return res

    def named_sub_kernels(self):
        for name, module in self.named_modules():
            if module is not self and isinstance(module, Kernel):
                yield name, module

    def num_outputs_per_input(self, x1, x2):
        """
        How many outputs are produced per input (default 1)
        if x1 is size `n x d` and x2 is size `m x d`, then the size of the kernel
        will be `(n * num_outputs_per_input) x (m * num_outputs_per_input)`
        Default: 1
        """
        return 1

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return exact_prediction_strategies.DefaultPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )

    def sub_kernels(self):
        for _, kernel in self.named_sub_kernels():
            yield kernel

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_

        # Check that ard_num_dims matches the supplied number of dimensions
        if settings.debug.on():
            if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
                raise RuntimeError(
                    "Expected the input to have {} dimensionality "
                    "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.size(-1))
                )

        if diag:
            res = super(Kernel, self).__call__(x1_, x2_, diag=True, last_dim_is_batch=last_dim_is_batch, **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diag()
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = lazify(super(Kernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params))
            return res

    def __getstate__(self):
        # JIT ScriptModules cannot be pickled
        self.distance_module = None
        return self.__dict__

    def __add__(self, other):
        kernels = []
        kernels += self.kernels if isinstance(self, AdditiveKernel) else [self]
        kernels += other.kernels if isinstance(other, AdditiveKernel) else [other]
        return AdditiveKernel(*kernels)

    def __mul__(self, other):
        kernels = []
        kernels += self.kernels if isinstance(self, ProductKernel) else [self]
        kernels += other.kernels if isinstance(other, ProductKernel) else [other]
        return ProductKernel(*kernels)

    def __setstate__(self, d):
        self.__dict__ = d

    def __getitem__(self, index):
        if len(self.batch_shape) == 0:
            return self

        new_kernel = deepcopy(self)
        # Process the index
        index = index if isinstance(index, tuple) else (index,)

        for param_name, param in self._parameters.items():
            new_kernel._parameters[param_name].data = param.__getitem__(index)
            ndim_removed = len(param.shape) - len(new_kernel._parameters[param_name].shape)
            new_batch_shape_len = len(self.batch_shape) - ndim_removed
            new_kernel.batch_shape = new_kernel._parameters[param_name].shape[:new_batch_shape_len]

        for sub_module_name, sub_module in self.named_sub_kernels():
            self._modules[sub_module_name] = sub_module.__getitem__(index)

        return new_kernel


class AdditiveKernel(Kernel):
    """
    A Kernel that supports summing over multiple component kernels.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> additive_kernel_matrix = covar_module(x1)
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all components are stationary.
        """
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels):
        super(AdditiveKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1, x2, diag=False, **params):
        res = ZeroLazyTensor() if not diag else 0
        for kern in self.kernels:
            next_term = kern(x1, x2, diag=diag, **params)
            if not diag:
                res = res + lazify(next_term)
            else:
                res = res + next_term

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)

        return new_kernel


class ProductKernel(Kernel):
    """
    A Kernel that supports elementwise multiplying multiple component kernels together.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) * RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> kernel_matrix = covar_module(x1) # The RBF Kernel already decomposes multiplicatively, so this is foolish!
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all components are stationary.
        """
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels):
        super(ProductKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1, x2, diag=False, **params):
        x1_eq_x2 = torch.equal(x1, x2)

        if not x1_eq_x2:
            # If x1 != x2, then we can't make a MulLazyTensor because the kernel won't necessarily be square/symmetric
            res = delazify(self.kernels[0](x1, x2, diag=diag, **params))
        else:
            res = self.kernels[0](x1, x2, diag=diag, **params)

            if not diag:
                res = lazify(res)

        for kern in self.kernels[1:]:
            next_term = kern(x1, x2, diag=diag, **params)
            if not x1_eq_x2:
                # Again delazify if x1 != x2
                res = res * delazify(next_term)
            else:
                if not diag:
                    res = res * lazify(next_term)
                else:
                    res = res * next_term

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)

        return new_kernel
    
    
def di_k(log_lda, log_eta, S1, S2, D2, L):
    
    lda = torch.exp(log_lda)
    eta = torch.exp(log_eta)
    
    k = (((1 + lda + eta)**(S2 - L/2))
    *((1 - lda + eta)**D2)
    *((1 + eta)**(S1 - L/2)) 
    * (1 - eta)**((L - S1 - S2 - D2)))
    
    return k


from gpytorch.constraints import Positive
from gpytorch.constraints import LessThan

class DiKernel(Kernel):
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

    def forward(self, geno1, geno2, **params):
        L = geno1.shape[1]
        geno1_ht = 1.*(geno1 == 1.)
        geno2_ht = 1.*(geno2 == 1.)        
        geno1_h0 = 1.*(geno1 == 0.)
        geno1_h1 = 1.*(geno1 == 2.)
        geno2_h0 = 1.*(geno2 == 0.)
        geno2_h1 = 1.*(geno2 == 2.)

        S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
        S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
        D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)

#         res = torch.exp(-.01*S1)*torch.exp(-.05*S2)*torch.exp(-.02*D2)*torch.exp(-.001*(L - S1 - S2 - D2))
#         return res

        return di_k(self.lda, self.eta, S1, S2, D2, L)



def rbf_k(length, S1, S2, D2):
    D1 = L - S1 - S2 - D2
    diff = (D1 + 4*D2)/L
    return torch.exp(-length*diff)

class RBFKernel(Kernel):
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

        
#     def forward(self, geno1, geno2, **params):
#         geno1_h0 = 1.*(geno1 == 0.)
#         geno1_h1 = 1.*(geno1 == 2.)
#         geno2_h0 = 1.*(geno2 == 0.)
#         geno2_h1 = 1.*(geno2 == 2.)

#         S1 = self.covar_dist(geno1, geno2, **params)
#         S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
#         D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)

# #         res = torch.exp(-.01*S1)*torch.exp(-.05*S2)*torch.exp(-.02*D2)*torch.exp(-.001*(L - S1 - S2 - D2))
# #         return res

#         return rbf_k(self.length, S1, S2, D2)

#     def calc_d(self, geno1, geno2, **params):
#         geno1_ht = 1.*(geno1 == 1.)
#         geno2_ht = 1.*(geno2 == 1.)
#         geno1_h0 = 1.*(geno1 == 0.)
#         geno1_h1 = 1.*(geno1 == 2.)
#         geno2_h0 = 1.*(geno2 == 0.)
#         geno2_h1 = 1.*(geno2 == 2.)

#         S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
#         S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
#         D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
#         D1 = L - S1 - S2 - D2
#         diff = (D1 + 4*D2)/L
#         return S1, S2, D2

#     def forward(self, geno1, geno2, diag=False, **params):
#         geno1_ht = 1.*(geno1 == 1.)
#         geno2_ht = 1.*(geno2 == 1.)
#         geno1_h0 = 1.*(geno1 == 0.)
#         geno1_h1 = 1.*(geno1 == 2.)
#         geno2_h0 = 1.*(geno2 == 0.)
#         geno2_h1 = 1.*(geno2 == 2.)

#         S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
#         S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
#         D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
        
#         return rbf_k(self.length, S1, S2, D2)

    def forward(self, geno1, geno2, diag=False, **params):
        geno1_ht = 1.*(geno1 == 1.)
        geno2_ht = 1.*(geno2 == 1.)
        geno1_h0 = 1.*(geno1 == 0.)
        geno1_h1 = 1.*(geno1 == 2.)
        geno2_h0 = 1.*(geno2 == 0.)
        geno2_h1 = 1.*(geno2 == 2.)

        S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
        S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
        D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
        D1 = L - S1 - S2 - D2
        diff = (D1 + 4*D2)/L

        return torch.exp(-diff)
    
    
# class LinKernel(Kernel):
#     """Additive kernel"""

#     is_stationary = True

#     # We will register the parameter when initializing the kernel
#     def __init__(self, 
#                 **kwargs):
#       super().__init__(**kwargs)

#     def forward(self, geno1, geno2, diag=False, **params):
#         geno1_h0 = 1.*(geno1 == 0.)
#         geno1_h1 = 1.*(geno1 == 2.)
#         geno2_h0 = 1.*(geno2 == 0.)
#         geno2_h1 = 1.*(geno2 == 2.)


#         S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
#         D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
#         diff = 2*(S2 - D2)

#         return diff



    
def d(geno1, geno2):
    """build distance tensor between two sets of genotypes
    geno1, geno2: n x L, m x L torch tensors

    """
    L = geno1.shape[1]
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


####### Basis kernel functions

def k_1_0(S1, S2, D1, D2):
    return -2*D2 + 2*S2

def k_0_1(S1, S2, D1, D2):
    return -2*D1 + L

def k_2_0(S1, S2, D1, D2):
    return 2*D2*(D2 - 1) - 4*D2*S2 + 2*(S2 -1)*S2

def k_1_1(S1, S2, D1, D2):
    return 2*(1 + 2*D1 -L)*(D2 - S2)

def k_0_2(S1, S2, D1, D2):
    return 2*D1**2 - 2*L*D1 + .5*(L-1)*L



class K1(Kernel):
    """Add + Dom kernel"""

    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, 
                par_prior=None, par_constraint=None, 
                **kwargs):
      super().__init__(**kwargs)

      # register the raw parameter
      self.register_parameter(
          name='raw_par', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 2))
      )

      # set the parameter constraint to be positive, when nothing is specified
      if par_constraint is None:
          par_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_par", par_constraint)


    # now set up the 'actual' paramter
    @property
    def par(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_par_constraint.transform(self.raw_par)
    @par.setter
    def par(self, value):
      return self._set_par(value)



    def forward(self, geno1, geno2, **params):

        global L
        L = geno1.shape[1]
        
        geno1_ht = 1.*(geno1 == 1.)
        geno2_ht = 1.*(geno2 == 1.)        
        geno1_h0 = 1.*(geno1 == 0.)
        geno1_h1 = 1.*(geno1 == 2.)
        geno2_h0 = 1.*(geno2 == 0.)
        geno2_h1 = 1.*(geno2 == 2.)

        S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
        S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
        D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
        D1 = L - S1 - S2 - D2

        return torch.exp(self.par[0])*k_1_0(S1, S2, D1, D2) + torch.exp(self.par[1])*k_0_1(S1, S2, D1, D2)

    

class K20(Kernel):
    """Pairwise epistatic"""

    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, 
                par_prior=None, par_constraint=None, 
                **kwargs):
      super().__init__(**kwargs)

      # register the raw parameter
      self.register_parameter(
          name='raw_par', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 2))
      )

      # set the parameter constraint to be positive, when nothing is specified
      if par_constraint is None:
          par_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_par", par_constraint)


    # now set up the 'actual' paramter
    @property
    def par(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_par_constraint.transform(self.raw_par)
    @par.setter
    def par(self, value):
      return self._set_par(value)


    def forward(self, geno1, geno2, **params):

        global L
        L = geno1.shape[1]
        
        geno1_ht = 1.*(geno1 == 1.)
        geno2_ht = 1.*(geno2 == 1.)        
        geno1_h0 = 1.*(geno1 == 0.)
        geno1_h1 = 1.*(geno1 == 2.)
        geno2_h0 = 1.*(geno2 == 0.)
        geno2_h1 = 1.*(geno2 == 2.)

        S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
        S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
        D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
        D1 = L - S1 - S2 - D2
        
        


        return torch.exp(self.par[0])*k_1_0(S1, S2, D1, D2) + torch.exp(self.par[1])*k_2_0(S1, S2, D1, D2)
    
    
class K2(Kernel):
    """Pairwise Epi + Dom"""

    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, 
                par_prior=None, par_constraint=None, 
                **kwargs):
      super().__init__(**kwargs)

      # register the raw parameter
      self.register_parameter(
          name='raw_par', 
          parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 5))
      )

      # set the parameter constraint to be positive, when nothing is specified
      if par_constraint is None:
          par_constraint = LessThan(upper_bound=0.)

      # register the constraint
      self.register_constraint("raw_par", par_constraint)


    # now set up the 'actual' paramter
    @property
    def par(self):
      # when accessing the parameter, apply the constraint transform
      return self.raw_par_constraint.transform(self.raw_par)
    @par.setter
    def par(self, value):
      return self._set_par(value)


    def forward(self, geno1, geno2, **params):

        global L
        L = geno1.shape[1]
        
        geno1_ht = 1.*(geno1 == 1.)
        geno2_ht = 1.*(geno2 == 1.)        
        geno1_h0 = 1.*(geno1 == 0.)
        geno1_h1 = 1.*(geno1 == 2.)
        geno2_h0 = 1.*(geno2 == 0.)
        geno2_h1 = 1.*(geno2 == 2.)

        S1 = self.covar_dist(geno1_ht, geno2_ht, **params)
        S2 = self.covar_dist(geno1_h0, geno2_h0, **params) + self.covar_dist(geno1_h1, geno2_h1, **params)
        D2 = self.covar_dist(geno1_h0, geno2_h1, **params) + self.covar_dist(geno1_h1, geno2_h0, **params)
        D1 = L - S1 - S2 - D2
        
        Ks = torch.stack([k_1_0(S1, S2, D1, D2), k_0_1(S1, S2, D1, D2), k_1_1(S1, S2, D1, D2), 
                    k_2_0(S1, S2, D1, D2), k_0_2(S1, S2, D1, D2)])
        Ks_reweighted = torch.mul(torch.exp(self.par).unsqueeze(1).unsqueeze(1), Ks)


        return torch.sum(Ks_reweighted, dim=0)
    
    