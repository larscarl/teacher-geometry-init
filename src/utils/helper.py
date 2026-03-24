import logging
import os
import sys

from pathlib import Path


def mb_to_bytes(mb):
    if mb is None:
        return -1
    if mb < 0:
        return -1
    return int(mb * 1024 * 1024)


def float_to_foldername(f, prefix=None, ndigits=4):
    s = f"{f:.{ndigits}f}".rstrip("0").rstrip(".")
    s = s.replace(".", "d")
    return f"{prefix}_{s}" if prefix else s


def configure_logging(log_path: Path) -> Path:
    """
    Initialise logging.
    If *log_path* is a directory (or has no suffix), create a log-file inside
    that directory called  <scriptname>_<timestamp>.log
    and return the final Path that is actually used.
    """
    import datetime as dt
    import logging
    import sys

    # Determine the file that should receive the logs
    if log_path.is_dir() or log_path.suffix == "":  # only a folder was given
        log_dir = log_path
        log_dir.mkdir(parents=True, exist_ok=True)
        script_name = Path(sys.argv[0]).stem
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")  # 20250714_103522
        log_file = log_dir / f"{script_name}_{ts}.log"
    else:  # full file path supplied
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path

    # Tell logging to write there
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
        force=True,  # overwrites any previous basicConfig call
    )
    logging.info("Logging started.")  # first line shows up immediately
    return log_file


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def is_rank0() -> bool:
    # torchrun sets RANK; if absent, assume single-process (rank 0)
    return os.environ.get("RANK", "0") == "0"


def setup_logging(name: str, level: str | None = "DEBUG") -> logging.Logger:
    """
    INFO/DEBUG -> stdout (captured by SLURM --output)
    WARNING+   -> stderr (captured by SLURM --error)
    Non-rank-0 workers only emit WARNING+ to reduce noise.
    """
    level = (level or os.environ.get("LMIC_LOGLEVEL", "DEBUG")).upper()
    root = logging.getLogger()
    # Clear any handlers Hydra/others may have set
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    format = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stderr handler: WARNING and above
    err_h = logging.StreamHandler(sys.stderr)
    err_h.setLevel(logging.WARNING)
    err_h.setFormatter(format)
    root.addHandler(err_h)

    if is_rank0():
        # stdout handler: up to INFO
        out_h = logging.StreamHandler(sys.stdout)
        out_h.setLevel(getattr(logging, level, logging.INFO))
        out_h.addFilter(MaxLevelFilter(logging.INFO))
        out_h.setFormatter(format)
        root.addHandler(out_h)
    else:
        # non-rank0: keep stderr only
        pass

    return logging.getLogger(name)
