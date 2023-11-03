import logging

logger = logging.getLogger(__name__)
logger.warning(
    "Module `fastsim.demos` has been imported. This should only "
    + "happen during testing, so you might be doing something wrong."
)