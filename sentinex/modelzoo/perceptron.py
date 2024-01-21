from sentinex.nn import Dense, Model

__all__ = ["Perceptron"]

class Perceptron(Model):
    """Perceptron is one of the first ANN ever to be introduced. It is a threshold logic unit
    that classifies inputs based on a fixed threshold.
    
    Arguments:
      activation (str or Activation): An activation function for the Perceptron to use.
    """
    def __init__(self,
                 activation="heaviside",
                 name="Perceptron",
                 *args,
                 **kwargs):
        super().__init__(
            name=name,
            *args,
            **kwargs
        )
        self.layer = Dense(1,
                           activation=activation)
    
    def call(self, x):
        return self.layer(x)

Perceptron.__module__ = "sentinex.modelzoo"