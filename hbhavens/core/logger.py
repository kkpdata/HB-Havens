import logging

def initialize_logger(filename='hbhavens.log', level=logging.INFO, include_console=False):

    handlers = [logging.FileHandler(filename, mode='w')]
    if include_console:
        handlers += [logging.StreamHandler()]
    
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s (%(name)s)',
        datefmt='%H:%M:%S',
        level=level,
        handlers=handlers
    )


class ProgressLogger:

    def __init__(self, logger, total, step):
        self.logger = logger
        self.total = total
        self.lastp = -1
        self.step = step
    
    def set_step(self, i):
        percentage = int(round(((i+1) / (self.total)) * 100))
        if percentage % self.step == 0:
            if self.lastp == percentage:
                return None
            self.lastp = percentage
            self.logger.info(f'Processing raster: {percentage:3d} %')

    



