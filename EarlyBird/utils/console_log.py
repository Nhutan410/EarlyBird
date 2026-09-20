import time

from lightning.pytorch.callbacks import Callback


class EpochConsoleLogger(Callback):
    """Print one plain line per epoch (losses, elapsed time, ETA).

    The default rich progress bar renders nothing when stdout is not a terminal (e.g. a
    subprocess inside a notebook) until fit() finishes, so long runs look frozen. This
    callback only prints; it does not touch training. Enable with
        --trainer.callbacks+=utils.console_log.EpochConsoleLogger
    """

    def on_fit_start(self, trainer, pl_module):
        self._t0 = time.time()
        self._start_epoch = trainer.current_epoch  # non-zero when resuming
        if not trainer.is_global_zero:  # under DDP only rank 0 prints
            return
        print(f'[epoch-log] fit start: epoch {trainer.current_epoch}/{trainer.max_epochs - 1}, '
              f'{trainer.num_training_batches} train batches/epoch', flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m = trainer.callback_metrics
        elapsed = time.time() - self._t0
        done = trainer.current_epoch - self._start_epoch + 1
        remaining = trainer.max_epochs - trainer.current_epoch - 1
        eta = elapsed / done * remaining
        train_loss = f"{m['train_loss']:.3f}" if 'train_loss' in m else 'n/a'
        val_loss = f"{m['val_loss']:.3f}" if 'val_loss' in m else 'n/a'
        print(f'[epoch-log] epoch {trainer.current_epoch}/{trainer.max_epochs - 1} done | '
              f'train_loss {train_loss} | val_loss {val_loss} | '
              f'elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min', flush=True)
