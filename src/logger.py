import logging
from rich.logging import RichHandler

logging.basicConfig(level="WARNING", format="%(message)s", handlers=[RichHandler(markup=True)])
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


def log(*vals):
    """Log values, highlighting any prefixed by a color tag (e.g., 'red:error')."""

    def fmt(v):
        for c in COLORS:
            tag = f"{c}:"
            if v.startswith(tag):
                return f"[bold {c}]{v[len(tag):]}[/bold {c}]"
        return v

    vals = list(map(str, vals))
    logger.info(" ".join(map(fmt, vals)) + ".")
