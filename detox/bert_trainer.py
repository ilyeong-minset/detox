import torch
from ignite.engine import Events

from detox.trainer import DetoxEngine, Trainer
from detox.utils import get_gradient_norm, get_parameter_norm


class BertDetoxEngine(DetoxEngine):
    """
    """

    def __init__(self, function, model, criterion, optimizer, scheduler, config):
        super().__init__(function, model, criterion, optimizer, config)

        self.scheduler = scheduler

    @staticmethod
    def train(engine, batch):
        """
        """
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = batch["input_ids"], batch["label_indices"]
        x, y = x.to(engine.device), y.to(engine.device)
        mask = batch["attention_mask"]
        mask = mask.to(engine.device)

        x = x[:, : engine.config.max_length]
        y_hat = engine.model(x, attention_mask=mask)[0]

        loss = engine.criterion(y_hat, y)
        loss.backward()

        # Calculate accuracy only if y is LongTensor, which means y is a one-hot representation
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        parameter_norm = float(get_parameter_norm(engine.model.parameters()))
        gradient_norm = float(get_gradient_norm(engine.model.parameters()))

        engine.optimizer.step()
        engine.scheduler.step()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "|param|": parameter_norm,
            "|grad|": gradient_norm,
        }

    @staticmethod
    def validate(engine, batch):
        """
        """
        engine.model.eval()

        with torch.no_grad():
            x, y = batch["input_ids"], batch["label_indices"]
            x, y = x.to(engine.device), y.to(engine.device)
            mask = batch["attention_mask"]
            mask = mask.to(engine.device)

            x = x[:, : engine.config.max_length]
            y_hat = engine.model(x, attention_mask=mask)[0]

            loss = engine.criterion(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            "loss": loss,
            "accuracy": accuracy,
        }


class BertTrainer(Trainer):
    """
    """

    def __init__(self, config):
        self.config = config

    def train(self, model, criterion, optimizer, scheduler, train_loader, valid_loader):
        training_engine = BertDetoxEngine(
            BertDetoxEngine.train, model, criterion, optimizer, scheduler, self.config
        )
        validation_engine = BertDetoxEngine(
            BertDetoxEngine.validate,
            model,
            criterion,
            optimizer,
            scheduler,
            self.config,
        )
        BertDetoxEngine.attach(
            training_engine, validation_engine, verbose=self.config.verbose
        )

        def run_validation(validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        training_engine.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, validation_engine, valid_loader
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, BertDetoxEngine.check_best
        )

        training_engine.run(train_loader, max_epochs=self.config.n_epochs)

        model.load_state_dict(validation_engine.best_model)

        return model
