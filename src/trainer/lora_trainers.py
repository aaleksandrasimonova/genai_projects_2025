import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class LoraTrainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, train_metrics: MetricTracker):
        """
        Run batch through the model, compute loss,
        and do training step.

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            train_metrics (MetricTracker): MetricTracker object that computes
                and aggregates training losses.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        # model forward, backward, opt step and etc.
        #
        # TO DO 
        # 
        # model_output = ...
        # batch.update(model_output)
        # losses = ...
        # batch.uodate(losses)
        

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            batch[loss_name] = batch[loss_name].mean()
            train_metrics.update(loss_name, batch[loss_name].item())

        return batch

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        # generate images via pipeline
        # TO DO
        # 
        # generated_images = ...
        # batch['generated'] = generated_images

        for metric in self.metrics:
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)
                
        return batch
        
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            prompt = batch['prompt']
            generated_img = batch['generated'][0].resize((256, 256))

            cutted_prompt = prompt.replace(" ", "_")[:30]
            image_name = f"{mode}_images/{cutted_prompt}"
            self.writer.add_image(image_name, generated_img)
