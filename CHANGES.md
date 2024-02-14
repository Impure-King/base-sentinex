<hr>

# Release 0.0.2-alpha

### Breaking Changes
(i) Removed ``compile`` method from ``sentinex.Module`` class.
(ii) ``Module.__call__`` is now jitted automatically in ``Module.__init__``, as long as the object has a ``__call__`` attribute.

### Compatible Major Changes
(i) Added ``tensorflow.data`` API for batch loading and removed ``jax_dataloader``, to improve training time.

### Performance Changes

### Minor Changes
(i) Added more docstrings indicating each module's arguments/function.

### Version Summary

<hr>