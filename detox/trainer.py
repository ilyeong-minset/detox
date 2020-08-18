from copy import deepcopy

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from detox.utils import get_gradient_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class DetoxEngine(Engine):
    """
    """

    def __init__(self, function, model, criterion, optimizer, config):
        super().__init__(function)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, batch):
        """
        """
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = batch.comment, batch.label
        x, y = x.to(engine.device), y.to(engine.device)

        x = x[:, : engine.config.max_length]
        y_hat = engine.model(x)

        loss = engine.criterion(y_hat, y)
        loss.backward()

        # Calculate accuracy only if y is LongTensor, which means that y is a one-hot representation
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        parameter_norm = float(get_parameter_norm(engine.model.parameters()))
        gradient_norm = float(get_gradient_norm(engine.model.parameters()))

        engine.optimizer.step()

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "|param|": parameter_norm,
            "|grad|": gradient_norm,
        }

    @staticmethod
    def validate(engine, batch):
        """
        """
        engine.model.eval()

        with torch.no_grad():
            x, y = batch.text, batch.label
            x, y = x.to(engine.device), y.to(engine.device)

            x = x[:, : engine.config.max_length]
            y_hat = engine.model(x)

            loss = engine.criterion(y_hat, y)

            # Calculate accuracy only if y is LongTensor, which means that y is a one-hot representation.
            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {"loss": float(loss), "accuracy": float(accuracy)}

    @staticmethod
    def attach(training_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        """
        """

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine, metric_name
            )

        training_metric_names = ["loss", "accuracy", "|param|", "|grad|"]
        for metric_name in training_metric_names:
            attach_running_average(training_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(training_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch
        if verbose >= VERBOSE_BATCH_WISE:

            @training_engine.on(Events.EPOCH_COMPLETED)
            def print_training_logs(engine):
                print(
                    f'Epoch {engine.state.epoch} >> |params|={engine.state.metrics["|param|"]:.2e} |grad|={engine.state.metrics["|grad|"]:.2e} loss={engine.state.metrics["loss"]:.4e} accuracy={engine.state.metrics["accuracy"]:.4f}'
                )

        validation_metric_names = ["loss", "accuracy"]
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch
        if verbose >= VERBOSE_BATCH_WISE:

            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_validation_logs(engine):
                print(
                    f'Validation >> loss={engine.state.metrics["loss"]:.4e} accuracy={engine.state.metrics["accuracy"]:.4f} best_loss={engine.best_loss:.4e}'
                )

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, training_engine, config, **kwargs):
        torch.save(
            {"model": engine.best_model, "config": config, **kwargs},
            config.model_function,
        )


class Trainer:
    """
    """

    def __init__(self, config):
        self.config = config

    def train(self, model, criterion, optimizer, train_loader, valid_loader):
        training_engine = DetoxEngine(
            DetoxEngine.train, model, criterion, optimizer, self.config
        )
        validation_engine = DetoxEngine(
            DetoxEngine.validate, model, criterion, optimizer, self.config
        )
        DetoxEngine.attach(
            training_engine, validation_engine, verbose=self.config.verbose
        )

        def run_validation(validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        training_engine.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, validation_engine, valid_loader
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, DetoxEngine.check_best
        )

        training_engine.run(train_loader, max_epochs=self.config.n_epochs)

        model.load_state_dict(validation_engine.best_model)

        return model
