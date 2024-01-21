# Sentinex - An (Experimental) Object Oriented Deep Learning Library Built on top of JAX.

Sentinex is a comprehensive deep learning library that aims to provide an intuitive *object oriented api* that is accelerated using JAX primitives.

Sentinex aims to provide a simplied and intuitive api that doesn't increase programming fatigue, when developing models. It offers low level abstractions like ``sx.Module``, while offering higher level subclasses like ``nn.Layers``,
``nn.Model``, ``nn.Activation``, ``nn.Initializers``, ``nn.Losses``, etc. Since everything is a PyTree, it is compatible with a wide variety of JAX ecosystem tools, like [Optax](https://optax.readthedocs.io/en/latest/index.html), [Equinox](https://docs.kidger.site/equinox/), [Keras](https://keras.io/about/), and so much more.

# Sharp Bits:
Currently, Sentinex is an immature framework that heavily utilizes external libraries for many core features. For example, Equinox is used to supply many of the filtered/lifted transformations, while Optax optimizers have been wrapped for extra-convenience.
This implies that Sentinex's internals are not maintained completely from this repo and is dependent on the support of other jax libraries. Therefore, there may be bugs in such interops, though Sentinex is aiming to prevent that and migrate to a more
independent status.
