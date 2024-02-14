from termcolor import colored

def batch_print(total_batches, current_batch, loss, metric=None, val_loss=None, val_metric=None):
  # Loading Handling:
  length = 20
  filled_length = int(length * current_batch // total_batches)
  bar = colored('─', "green") * filled_length + \
    colored('─', "yellow") * (length - filled_length)

  # Variable handling:
  metric = metric or loss
  if val_loss is None:
    val_loss_str = ""
  else:
    val_loss_str = f" - val_loss: {val_loss:>.4f}"

  if val_metric is None:
    val_met_str = ""
  else:
    val_met_str = f" - val_metrics: {val_metric:>.4f}"
  print(f'\rBatch {current_batch}/{total_batches} {bar} - loss: {loss:>.4f} - metric: {metric:>.4f}' + val_loss_str + val_met_str, end='', flush=True)