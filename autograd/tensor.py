from turtle import backward
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    
    return np.array(arrayable)

class Tensor:
    data: np.ndarray
    depends_on: List[Dependency]

    def __init__(self, data:np.ndarray, 
                 requires_grad: bool = False,
                 depends_on = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or [] 
        self.shape = self.data.shape
        self.grad: Optional[Tensor] = None

        if self.requires_grad:
            self.zero_grad()
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor" 


        # This is for top level grad handling. 
        # For the first step in the back prop, the
        # grad should be default to 1 for zero tensors
        # Imagine f: R^n -> R, we set d_/df = 1
        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data # Remember it should alwasys be accumulated, also need be reset after

        for dependency in self.depends_on: # This is doing DFS, does this really work? (Yes, it works, try to plot it and verify, notice we do grad.data instead of the accumulated self.grad.data when we calculate the dependency grad)
            backward_grad = dependency.grad_fn(grad.data) # NOTE: self.grad.data? (No, it is doing DFS, should only use the back prop flow from this stream)
            dependency.tensor.backward(Tensor(backward_grad))
        
    def sum(self) -> 'Tensor':
        return tensor_sum(self)

def tensor_sum(t: Tensor) -> Tensor:
    """
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            Refer to the notes in the readling list.
            >>> t = [5, 6, 7, 8]
            >>> t2 = t.sum()
            """
            # It's doing the reverse mode back propagation. 
            # Which means each time we can only handles one scalar function f_i. 
            # Then sum() returns an one-dimentional result, thus the grad are expected 
            # to be a 1x1 vector (scalar). 
            # But with the broadcasting of numpy array, suppose f: R^n -> R^m
            # the grad can be a m x 1 vector, results in a perfect calculation with boradcasting
            # If it's the sum, each time it could only have one
            return grad * np.ones_like(t.data)
        
        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)