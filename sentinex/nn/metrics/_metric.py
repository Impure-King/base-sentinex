from ...module import Module

__all__ = ["Metric"]

class Metric(Module):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
    
    def call(self, *args, **kwargs):
        return NotImplementedError