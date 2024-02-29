from typing import Optional
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import optax
from termcolor import colored
from ..optimizers.base_optimizers import OptaxOptimizer

from sentinex.module import Module

__all__ = ["TrainState"]


class TrainState(Module):
    def __init__(self,
                 model,
                 name="TrainState",
                 *args,
                 **kwargs):
        super().__init__(name,
                         *args,
                         **kwargs)
        self.model = model

    def compile(
        self,
        loss_fn,
        optimizer: OptaxOptimizer,
        metric_fn=None
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn or loss_fn

        def grad_fn(model,
                    X,
                    y):
            pred = model(X, training=True)
            return jnp.mean(self.loss_fn(y, pred)), pred

        self.grad_fn = jax.jit(jax.value_and_grad(grad_fn, has_aux=True))
        self.jit_train_step = jax.jit(self.train_step)

    def train_step(self,
                   model,
                   X,
                   y):
        (loss, pred), grads = self.grad_fn(model, X, y)
        model = self.optimizer.apply_gradients(grads, model)
        return model, loss, pred

    def fit(self,
            X_train,
            y_train,
            epochs=1,
            batch_size=None,
            train_shuffle=True,
            train_drop_last=False,
            validation_data: Optional[list] = None,
            validation_batch_size = None,
            validation_shuffle = False,
            validation_drop_last = False,
            verbosity=1,
            val_data_loader=None):
        # Batching the data:
        train_dataset = jdl.ArrayDataset(X_train, y_train)
        batch_size = batch_size or (len(train_dataset) if len(train_dataset) < 32 else 32)
        
        train_data_loader = jdl.DataLoader(train_dataset,
                                           backend="jax",
                                           batch_size=batch_size,
                                           shuffle=train_shuffle,
                                           drop_last=train_drop_last)
        
        train_batch_no = len(train_data_loader)
        anim = self.verbosity_setter(verbosity)
        anim_step = train_batch_no//30
        
        if validation_data:
            valid_dataset = jdl.ArrayDataset(*validation_data)
            validation_batch_size = validation_batch_size or (len(valid_dataset) if len(valid_dataset) < 32 else 32)
            valid_data_loader = jdl.DataLoader(valid_dataset,
                                               backend='jax',
                                               batch_size=validation_batch_size,
                                               shuffle=validation_shuffle,
                                               drop_last=validation_drop_last)
            valid_batch_no = len(valid_data_loader)
            valid_anim_step = train_batch_no//30
        # Training:
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}:")
            for batch_no, (X, y) in enumerate(train_data_loader):
                self.model, loss_val, pred = self.jit_train_step(self.model, X, y)
                metric_val = self.metric_fn(y, pred)
                if (batch_no + 1) % anim_step == 0:
                    anim(train_batch_no, batch_no + 1, loss_val, metric_val)
            print('')
            if validation_data:
                for batch_no, (X, y) in enumerate(valid_data_loader):
                    pred = self.model(X, training=False)
                    loss_val = self.loss_fn(y, pred)
                    metric_val = self.metric_fn(y, pred)
                    if (batch_no + 1) % valid_anim_step == 0:
                        anim(valid_batch_no, batch_no + 1, loss_val, metric_val)
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
        length = 20
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

    def evaluate(self, X_test, y_test):
        pred = self.model(X_test)
        return self.loading_animation(1, 1, self.loss_fn(y_test, pred), self.metric_fn(y_test, pred))
    
    def return_model(self):
        return self.model


TrainState.__module__ = 'sentinex.nn'
