import time
from datetime import timedelta
from dataclasses import dataclass, field

@dataclass(eq=False)
class EarlyStopping:
    patience: int = field(
        default=None, metadata={"help": "how many epochs to wait before stopping when loss is not improving"}
    )
    min_delta: int = field(
        default=None, metadata={"help": "minimum difference between new loss and old loss for new loss to be considered as an improvement"}
    )
        
    def __post_init__(self):
        self.counter = 0
        self.best_loss = None
        self.status = False

    def __call__(self, monitor_loss, epoch):
        if self.best_loss == None:
            self.best_loss = monitor_loss
        elif self.best_loss - monitor_loss > self.min_delta:
            self.best_loss = monitor_loss
        elif self.best_loss - monitor_loss <= self.min_delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                print('Early stopping at epoch {} .'.format(epoch+1))
                self.status = True


@dataclass(eq=False)
class TimeStopping:
    time_stopping_seconds: int = field(
        default=None, metadata={"help": "total wait traing seconds"}
    )
    
    def __post_init__(self):
        self.start_train_time = time.time()
        self.stopping_train_time = self.start_train_time + self.time_stopping_seconds
        self.status = False

    def __call__(self, current_time, epoch):
        if current_time >= self.stopping_train_time:
            diff_seconds = time.time() - self.start_train_time
            formatted_diff_seconds = timedelta(seconds=diff_seconds)
            print('Timed stopping at epoch {} after training for {}s'.format(
                    epoch+1, formatted_diff_seconds))
            self.status = True

@dataclass(eq=False)
class AugmentCallbacks:
    callback_parameters: field(default_factory=dict)
    
    def __post_init__(self):
        self.callback_stop_status = False
        
        if self.callback_parameters['use_early_stopping']:
            self.early_stopping_callback = EarlyStopping(
                patience=self.callback_parameters['early_stopping_patience'],
                min_delta=self.callback_parameters['early_stopping_min_delta'])

        if self.callback_parameters['use_time_stopping']:
            self.time_stopping_callback = TimeStopping(
                time_stopping_seconds=self.callback_parameters['time_stopping_seconds'])

    def __call__(self, **kwargs):
        overall_callbacks = []

        if self.callback_parameters['use_early_stopping']:
            self.early_stopping_callback(
                monitor_loss=kwargs['monitor_loss'],
                epoch=kwargs['epoch'])

            overall_callbacks.append(self.early_stopping_callback.status)

        if self.callback_parameters['use_time_stopping']:
            self.time_stopping_callback(
                current_time=time.time(),
                epoch=kwargs['epoch'])

            overall_callbacks.append(self.time_stopping_callback.status)

        if any(overall_callbacks):
            self.callback_stop_status = True
