import os
from omegaconf import DictConfig
from hydra.experimental.callback import Callback

class SavePathExistsCallback(Callback):
    def on_run_start(self, config: DictConfig, **kwargs: any) -> None:
        if os.path.exists(config.hydra.run.dir):
            raise BaseException(f"Output dir already exists! Current output path is: {config.hydra.run.dir}")

    def on_multirun_start(self, config: DictConfig, **kwargs: any) -> None:
        if os.path.exists(os.path.join(config.hydra.sweep.dir, config.hydra.sweep.subdir)):
            raise BaseException("Output dir already exists! Current output path is: " +
                                f"{os.path.join(config.hydra.sweep.dir, config.hydra.sweep.subdir)}")
        