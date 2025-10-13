import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging_inference
import datetime


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_inference_lora")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    set_random_seed(config.inferencer.seed)

    project_config = OmegaConf.to_container(config)
    
    logger = setup_saving_and_logging_inference(config)
    writer = instantiate(config.writer, logger, project_config)

    device = torch.device(config.trainer.device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    # build model architecture, then print to console
    model = instantiate(config.model, device=device)
    model.prepare_for_training()


    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    
    pipeline = instantiate(
        config.pipeline,
        model=model,
        device=device
    )
        
    inferencer = instantiate(
        config.inferencer,
        model=model,
        pipe=pipeline,
        metrics=metrics,
        global_config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False
    )

    inferencer.inference()


if __name__ == "__main__":
    main()
