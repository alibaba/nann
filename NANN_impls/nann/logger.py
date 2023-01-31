import logging
import os
import sys

from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
  def __init__(self, *args, **kwargs):
    self._root_name = kwargs.pop("root_name") + "."
    self._abbrev_name = kwargs.pop("abbrev_name", "")
    if len(self._abbrev_name):
      self._abbrev_name = self._abbrev_name + "."
    super(_ColorfulFormatter, self).__init__(*args, **kwargs)

  def formatMessage(self, record):
    record.name = record.name.replace(self._root_name, self._abbrev_name)
    log = super(_ColorfulFormatter, self).formatMessage(record)
    if record.levelno == logging.WARNING:
      prefix = colored("WARNING", "red", attrs=["blink"])
    elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
      prefix = colored("ERROR", "red", attrs=["blink", "underline"])
    else:
      return log
    return prefix + " " + log


def get_logger(name, output=None, color=True):
  plain_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(pathname)s:%(lineno)d] %(funcName)s: %(message)s",
    datefmt="%m/%d %H:%M:%S"
  )

  colored_formatter = _ColorfulFormatter(
    colored("[%(asctime)s] [%(levelname)s] [%(pathname)s:%(lineno)d] %(funcName)s: ", "green") + "%(message)s",
    datefmt="%m/%d %H:%M:%S",
    root_name=name,
    abbrev_name=str(name),
  )

  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)

  ch = logging.StreamHandler(stream=sys.stdout)
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(colored_formatter if color else plain_formatter)
  logger.addHandler(ch)

  # file logging
  if output is not None:
    if output.endswith(".txt") or output.endswith(".log"):
      filename = output
    else:
      filename = os.path.join(output, "log.txt")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.StreamHandler(open(filename, "a"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)

  return logger
