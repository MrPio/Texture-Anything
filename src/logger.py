import logging
from rich.logging import RichHandler
from termcolor import colored

logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(markup=True, show_path=False)])
logger = logging.getLogger("src")

COLORS = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "light_grey",
    "dark_grey",
    "light_red",
    "light_green",
    "light_yellow",
    "light_blue",
    "light_magenta",
    "light_cyan",
]


def log(*vals, use_log=True):
    """Log values, highlighting any prefixed by a color tag (e.g., 'red:error')."""

    def fmt(v):
        v = str(v)
        for c in COLORS:
            tag = f"{c}:"
            if v.startswith(tag):
                return f"[bold {c}]{v[len(tag):]}[/bold {c}]" if use_log else colored(v[len(tag) :], c, attrs=["bold"])
        return v

    vals = map(fmt, vals)
    if use_log:
        logger.info(" ".join(vals) + ".")
    else:
        print(*vals)


def cprint(*vals):
    log(*vals, use_log=False)
