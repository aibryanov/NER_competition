import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Global random seed set to %d", seed)
