import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import optax
from termcolor import colored

from sentinex.module import Module

__all__ = ["TrainState"]


class TrainState(Module):
    def __init__(self,
                 model,
                 name="TrainState",
                 trainable=False,
                 *args,
                 **kwargs):
        super().__init__(name,
                         trainable=trainable,
                         *args,
                         **kwargs)
        self.model = model

    def compile(
        self,
        loss,
        optimizer,
        metrics=None
    ):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics or loss
        self.state = optimizer.init(self.model)

        def grad_fn(model,
                    X,
                    y):
            pred = model(X)
            return jnp.mean(self.loss(y, pred)), pred

        self.grad_fn = jax.jit(jax.value_and_grad(grad_fn, has_aux=True))
        self.freeze("grad_fn")
        self.train_step = jax.jit(self.train_step)

    def train_step(self,
                   model,
                   X,
                   y):
        (loss, pred), grads = self.grad_fn(model, X, y)
        updates, self.state = self.optimizer.update(grads, self.state, model)
        model = optax.apply_updates(model, updates)
        return model, loss, pred

    def fit(self,
            X_train,
            y_train,
            epochs=1,
            batch_size=None,
            train_shuffle=True,
            train_drop_last=False,
            verbosity=1,
            val_data_loader=None):
        # Batching the data:
        train_dataset = jdl.ArrayDataset(X_train, y_train)
        batch_size = batch_size or (
            len(train_dataset) if len(train_dataset) < 32 else 32)
        train_data_loader = jdl.DataLoader(train_dataset,
                                           backend="jax",
                                           batch_size=batch_size,
                                           shuffle=train_shuffle,
                                           drop_last=train_drop_last)
        total_batch_no = len(train_data_loader)
        anim = self.verbosity_setter(verbosity)
        # train_step = jax.jit(self.train_step)
        # Training:
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}:")
            for batch_no, (X, y) in enumerate(train_data_loader):
                self.model, loss_val, pred = self.train_step(self.model, X, y)
                metric_val = self.metrics(y, pred)
                anim(total_batch_no, batch_no + 1, loss_val, metric_val)
            print('\n')

    def verbosity_setter(self, verbosity):
        if verbosity == 0:
            return self.loading_animation_1
        elif verbosity == 1:
            return self.loading_animation
        else:
            raise ValueError("Verbosity can only be 0 or 1.")

    def loading_animation_1(self, total_batches, current_batch, loss, metric, val_loss=None, val_metric=None):
        print(f"\r Batch {current_batch}/{total_batches} \t\t Loss: {loss:>0.2f} \t\t Metrics: {metric:>0.2f}", end='', sep='')

    def loading_animation(self, total_batches, current_batch, loss, metric, val_loss=None, val_metric=None):
        length = 30
        filled_length = int(length * current_batch // total_batches)
        bar = colored('─', "green") * filled_length + \
            colored('─', "yellow") * (length - filled_length)
        if val_loss is None:
            val_loss_str = ""
        else:
            val_loss_str = f" - val_loss: {val_loss:>.2f}"

        if val_metric is None:
            val_met_str = ""
        else:
            val_met_str = f" - val_metrics: {val_metric:>.2f}"
        print(f'\rBatch {current_batch}/{total_batches} {bar} - loss: {loss:>.2f} - metric: {metric:>.2f}' + val_loss_str + val_met_str, end='', flush=True)

    def return_model(self):
        return self.model


TrainState.__module__ = 'sentinex.nn'
