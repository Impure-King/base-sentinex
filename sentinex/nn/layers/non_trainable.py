from sentinex.nn.layers.base_layer import Layer


class Flatten(Layer):
    def __init__(self, name="Flatten", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    
    def call(self, x):
        return x.reshape((x.shape[0], -1))