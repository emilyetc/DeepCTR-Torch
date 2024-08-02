import torch
import os

class ModelCheckpoint:
    """Save the PyTorch model after every epoch.
    
    `filepath` can contain named formatting options, which will be filled with the value of `epoch` and keys in `logs` (passed in `on_epoch_end`).
    
    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, the latest best model according to the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}. If `save_best_only=True`, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For `val_acc`, this should be `max`, for `val_loss` this should be `min`, etc. In `auto` mode, the direction is automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be saved (`model.state_dict()`), else the full model is saved using `torch.save`.
        period: Interval (number of epochs) between checkpoints.
    """
    
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', save_weights_only=False, period=1):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.monitor_op = min if mode == 'min' else max
        
        if mode == 'auto':
            if 'acc' in self.monitor:
                self.monitor_op = max
                self.best = -float('inf')
            else:
                self.monitor_op = min
                self.best = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    if self.verbose > 0:
                        print('Can save best model only with {} available, skipping.'.format(self.monitor))
                else:
                    if self.monitor_op(current, self.best) != self.best:
                        if self.verbose > 0:
                            print('Epoch {:05d}: {} improved from {:.5f} to {:.5f}, saving model to {}'.format(epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        self._save_model(epoch, filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch {:05d}: {} did not improve from {:.5f}'.format(epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('Epoch {:05d}: saving model to {}'.format(epoch + 1, filepath))
                self._save_model(epoch, filepath)
    
    def _save_model(self, epoch, filepath):
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if self.save_weights_only:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.model, filepath)
