from datetime import datetime
import numpy as np
import pandas as pd


class ClearMLWriter:
    """
    Class for experiment tracking via ClearML.

    See https://clear.ml/
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        task_name=None,
        tags=None,
        **kwargs,
    ):
        """
        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            task_name (str | None): name of the task/run. If None, random name
                is given.
            tags (list | None): list of tags for the experiment.
        """
        try:
            from clearml import Task

            # Инициализируем задачу в ClearML
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name or f"SDXL-LoRA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags=tags,
                reuse_last_task_id=True
            )
            
            # Логируем конфигурацию эксперимента
            self.task.connect(project_config)
            
            self.logger = logger
            self.logger.info(f"ClearML initialized: {self.task.get_task_id()}")

        except ImportError:
            logger.warning("For use ClearML install it via: \n\t pip install clearml")
            self.task = None

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "general/steps_per_sec", 
                (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        if self.task is None:
            return
            
        from clearml import Logger
        Logger.current_logger().report_scalar(
            title=scalar_name,
            series=self.mode,
            value=scalar,
            iteration=self.step
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        if self.task is None:
            return
            
        from clearml import Logger
        for scalar_name, scalar_value in scalars.items():
            Logger.current_logger().report_scalar(
                title=scalar_name,
                series=self.mode,
                value=scalar_value,
                iteration=self.step
            )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | ndarray | Image): image in the ClearML-friendly format.
        """
        if self.task is None:
            return
            
        from clearml import Logger
        
        # Конвертируем PIL Image в numpy array если нужно
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            # Если это путь к файлу
            image_np = image
            
        Logger.current_logger().report_image(
            title=image_name,
            series=self.mode,
            image=image_np,
            iteration=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (Path | ndarray): audio in the ClearML-friendly format.
            sample_rate (int): audio sample rate.
        """
        if self.task is None:
            return
            
        # ClearML пока не имеет прямой поддержки аудио через Logger
        # Но можно залогировать как файл
        try:
            from clearml import Logger
            if hasattr(audio, 'detach'):
                audio = audio.detach().cpu().numpy()
            
            # Сохраняем временный файл и логируем его
            import tempfile
            import scipy.io.wavfile as wavfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                if sample_rate is None:
                    sample_rate = 22050
                wavfile.write(f.name, sample_rate, audio)
                Logger.current_logger().report_media(
                    title=audio_name,
                    series=self.mode,
                    local_path=f.name,
                    iteration=self.step
                )
        except Exception as e:
            self.logger.warning(f"Could not log audio: {e}")

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
        if self.task is None:
            return
            
        from clearml import Logger
        Logger.current_logger().report_text(
            f"{text_name}: {text}",
            print_console=False
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.

        Args:
            hist_name (str): name of the histogram to use in the tracker.
            values_for_hist (Tensor): array of values to calculate
                histogram of.
            bins (int | str): the definition of bins for the histogram.
        """
        if self.task is None:
            return
            
        from clearml import Logger
        values_for_hist = values_for_hist.detach().cpu().numpy()
        
        Logger.current_logger().report_histogram(
            title=hist_name,
            series=self.mode,
            values=values_for_hist,
            iteration=self.step,
            xlabels=bins
        )

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker.

        Args:
            table_name (str): name of the table to use in the tracker.
            table (DataFrame): table content.
        """
        if self.task is None:
            return
            
        from clearml import Logger
        Logger.current_logger().report_table(
            title=table_name,
            series=self.mode,
            table_plot=table,
            iteration=self.step
        )

    def add_images(self, images_name, images):
        """
        Log multiple images to the experiment tracker.

        Args:
            images_name (str): base name for the images.
            images (list): list of images.
        """
        if self.task is None:
            return
            
        for i, image in enumerate(images):
            self.add_image(f"{images_name}_{i}", image)

    def close(self):
        """
        Close the ClearML task.
        """
        if self.task is not None:
            self.task.close()