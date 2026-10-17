from omegaconf import DictConfig

from src.training import Trainer
from src.estimator import build_model


def train(cfg: DictConfig) -> None:

    device = 'cuda' if cuda.is

    model = build_model(cfg=cfg)
    trainer = Trainer(model=model, loss_function=loss_function, )
