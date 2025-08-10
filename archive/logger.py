import logging

class Colors:
    # Regular colors
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # Bright colors
    BRIGHT_BLACK   = "\033[90m"
    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    # Reset
    RESET = "\033[0m"

class Logger:
    """
    Custom Logger class that wraps Python's built-in logging module.
    Provides a simple interface for logging messages at various severity levels.
    """

    def __init__(
        self,
        name: str = "app",
        level: int = logging.INFO,
        format: str = "[%(asctime)s] %(levelname)s: %(message)s",
        force: bool = True,
    ):
        """
        Initialize the Logger instance.

        Args:
            name (str): Name of the logger.
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        # Set up the root logger configuration (optional, only once per app)
        logging.basicConfig(
            level=level,
            format=format,
            force=force
        )

        self.logger = logging.getLogger(name)
        self.logger.propagate = False

        # Add a stream handler if there are no handlers
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(f"{Colors.BRIGHT_BLUE}{msg}{Colors.RESET}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(f"{Colors.YELLOW}{msg}{Colors.RESET}", *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(f"{Colors.RED}{msg}{Colors.RESET}", *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(f"{Colors.RED}{msg}{Colors.RESET}", *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        self.logger.exception(f"{Colors.RED}{msg}{Colors.RESET}", *args, **kwargs)