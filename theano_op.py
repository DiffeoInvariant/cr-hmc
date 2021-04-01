from crikit.cr import PointMap, JAXArrays
from crikit.invariants import TensorType
from pyadjiont_utils import array, ndarray, overload_jax, ReducedFunction, Control
from pyadjoint import ReducedFunctional, get_working_tape
from jax import numpy as jnp
import jax
from typing import Tuple, Iterable
import logging
#from math import prod
import numpy as np
import theano
#Theano to interface to PyMC3
import theano.tensor as tt
import pymc3 as pm
#theano.config.compute_test_value = 'off'



def flatten_param(param):
    desc = param.shape
    flat = param.flatten()
    return desc, flat


def unflatten_param(desc, flat):
    return jnp.reshape(jnp.array(np.array(flat)),desc)


def get_shape(x):
    try:
        return x.shape
    except AttributeError:
        return ()


def flatten_params(params):
    """Flatten parameters (of possibly different shapes) into a single list of floats
    
    :param params: The parameters to flatten
    :type params: pytree
    :return: A tuple ``(desc, flats)`` where ``desc`` is a description that can
        be used in conjunction with :func:`unflatten_params` to get back the 
        original pytree.
    :rtype: tuple

    """
    jax_flats, treedef = jax.tree_util.tree_flatten(params)
    shape_defs = [get_shape(x) for x in jax_flats]
    true_flats = []
    flat_descs = [None] * len(params)
    for i,par in enumerate(params):
        desc, flat = flatten_param(par)
        true_flats += list(flat)
        flat_descs[i] = desc

    return (treedef, flat_descs),true_flats



def unflatten_params(desc, flat_params):

    treedef, flat_descs = desc
    jax_flats = [None] * len(flat_descs)
    offset = 0
    for i,descr in enumerate(flat_descs):
        nelem = int(np.prod(descr))
        elements = flat_params[offset:(offset+nelem)]
        jax_flats[i] = unflatten_param(descr,elements)
        offset += nelem

    return jax.tree_util.tree_unflatten(treedef,jax_flats)

    



class TheanoLogPdf(tt.Op):


    itypes = [tt.dvector] # parameters are flattened into a list of floats
    otypes = [tt.dscalar]

    def __init__(self, rf, params, profile_memory=False):
        """

        :param rf: a :class:`pyadjoint_utils.ReducedFunction` that outputs the log-pdf of the model.
        :type rf: pyadjoint_utils.ReducedFunction
        :param params: an example of the parameters (``Control``s) of ``rf``
        :type params: Iterable[Any]
        :param profile_memory: if True, uses the 
            :ref:`jax memory profiler <https://jax.readthedocs.io/en/latest/device_memory_profiling.html>` to profile the memory usage of the ``Op``, defaults to False
        :type profile_memory: bool, optional
        """
        self.rf = rf
        self.descr, self.flats = flatten_params(params)
        self.controls = self.rf.controls
        self.logpgrad = TheanoLogPdfGrad(rf,self.descr, self.rf.controls)
        self.profile_memory = profile_memory
        self.profile_count = 0
        self.grad_profile_count = 0
        if profile_memory:
            import jax.profiler

    def perform(self, node, inputs, outputs):
        
        (theta,) = inputs
        params = unflatten_params(self.descr,np.reshape(theta,(theta.size,)))
        for i in range(len(params)):
            self.controls[i].update(params[i])

        print(f"Theta: {params}")
        loglik = self.rf(params)
        
        if isinstance(loglik,ndarray):
            loglik = loglik.unwrap(True)
            
        outputs[0][0] = np.array(loglik)
        if self.profile_memory:
            jax.profiler.save_device_memory_profile(f"memory_fwd_{self.profile_count}.prof")
            self.profile_count += 1


    def grad(self, inputs, g):
        #calculates the vjp of the log-pdf w.r.t the parameters in g[0]
        (theta,) = inputs
        grad = self.logpgrad(theta)
        val = [g[0] * grad]
        if self.profile_memory:
            jax.profiler.save_device_memory_profile(f"memory_adj_{self.grad_profile_count}.prof")
            self.grad_profile_count += 1
        return val
        
        
        

class TheanoLogPdfGrad(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dvector]


    def __init__(self, rf, descr, ctrls):
        self.descr = descr
        self.rf = rf
        self.logpdf_rf = None
        self.controls = ctrls
        

    
    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        params = unflatten_params(self.descr,theta)
        
        for i in range(len(params)):
            self.rf.controls[i].update(params[i])

        grad = self.rf.derivative()
        if not isinstance(grad,(list,tuple)):
            grad = [grad]
        
        gval = [x.unwrap(True) for x in grad]
        gdesc, gflat = flatten_params(gval)
        outputs[0][0] = np.array(gflat,dtype=float)
        


