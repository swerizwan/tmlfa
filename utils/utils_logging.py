import logging  # For logging messages
import os  # For interacting with the operating system
import sys  # For accessing command-line arguments and system-specific parameters


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.

    This class is useful for tracking metrics like loss or accuracy during training.
    It provides methods to reset the meter, update it with new values, and compute the average.
    """

    def __init__(self):
        """
        Initialize the AverageMeter with None values.
        """
        self.val = None  # Current value
        self.avg = None  # Average value
        self.sum = None  # Sum of all values
        self.count = None  # Number of values
        self.reset()  # Reset the meter to initial state

    def reset(self):
        """
        Reset the meter to its initial state.
        """
        self.val = 0  # Reset current value to 0
        self.avg = 0  # Reset average value to 0
        self.sum = 0  # Reset sum to 0
        self.count = 0  # Reset count to 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The new value to update the meter with.
            n (int, optional): The number of times the value should be counted. Defaults to 1.
        """
        self.val = val  # Update the current value
        self.sum += val * n  # Update the sum
        self.count += n  # Update the count
        self.avg = self.sum / self.count  # Recalculate the average


def init_logging(rank, models_root):
    """
    Initialize logging for the training process.

    Args:
        rank (int): The rank of the process (used for distributed training).
        models_root (str): The root directory where model logs and checkpoints will be saved.
    """
    if rank == 0:  # Only initialize logging for the main process (rank 0)
        log_root = logging.getLogger()  # Get the root logger
        log_root.setLevel(logging.INFO)  # Set the logging level to INFO

        # Define a formatter for the log messages
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")

        # Create a file handler to write logs to a file
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)  # Set the formatter for the file handler

        # Create a stream handler to write logs to the console
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_stream.setFormatter(formatter)  # Set the formatter for the stream handler

        # Add the handlers to the root logger
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)

        # Log the rank of the process
        log_root.info('rank_id: %d' % rank)