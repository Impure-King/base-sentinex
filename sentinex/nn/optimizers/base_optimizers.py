from typing import Optional
from sentinex.module import Module
from jax.tree_util import tree_map
from equinox import filter_jit, partition, is_array, combine, filter
import optax

class Optimizer(Module):
    def __init__(self, name: str = "Optimizer",
                 **kwargs) -> None:
        super().__init__(name=name,
                         **kwargs)
        # self.apply_gradients = filter_jit(self.apply_gradients)
    
    def update_rule(self, grads, params):
        return NotImplementedError
    
    @filter_jit
    def apply_gradients(self, grads, model):
        params, static = partition(model, is_array)
        grad_params, grad_static = partition(grads, is_array)
        return combine(tree_map(self.update_rule, grad_params, params), static)

class SGD(Optimizer):
    velocity: float
    def __init__(self, learning_rate: float = 1e-2,
                 momentum:float = 0.0,
                 nesterov: bool = False,
                 name: str = "SGD") -> None:
        super().__init__(name)
        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.momentum = momentum
        self.velocity = 0
    
    def update_rule(self, grads, params):
        """Note the grads are put first because it allows the tree_map to use the grad structure instead of the params
        and avoid any jax tree_map conflicts.

        Args:
            grads (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        
        if not self.nesterov:
            params += self.velocity
        else:
            params = params + self.momentum * params - self.learning_rate * grads
        return params

class OptaxOptimizer(Optimizer):
    """A Wrapper around Optax optimizers for convenience.

    Args:
        Module (_type_): _description_
    """
    def __init__(self, optimizer, params, name: str = "OptaxOptimizer") -> None:
        super().__init__(name=name,
                         dynamic=True,
                         )
        self.optimizer = optimizer
        self.init(params)
    
    def init(self, model):
        params = filter(model, is_array)
        self.state = self.optimizer.init(params)
    
    def apply_gradients(self, grads, model):
        params, static = partition(model, is_array)
        grad_params = filter(grads, is_array)
        updates, self.state = self.optimizer.update(grad_params, self.state, params)
        updated_params = optax.apply_updates(params, updates)
        model = combine(updated_params, static)
        return model

class OptaxSGD(OptaxOptimizer):
    def __init__(self,
                 params,
                 learning_rate:float = 1e-2,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 name: str  = "OptaxSGD") -> None:
        optimizer = optax.sgd(learning_rate=learning_rate,
                              momentum=momentum,
                              nesterov=nesterov)
        super().__init__(optimizer, params, name)

class OptaxAdam(OptaxOptimizer):
    def __init__(self,
                 params,
                 learning_rate:float = 1e-2,
                 b1: float = 0.9,
                 b2:float = 1 - 1e-3,
                 eps:float = 1e-8,
                 eps_root: float = 0,
                 name: str = "OptaxAdam") -> None:
        optimizer = optax.adam(learning_rate=learning_rate,
                               b1=b1,
                               b2=b2,
                               eps=eps,
                               eps_root=eps_root)
        super().__init__(optimizer, params, name)
        