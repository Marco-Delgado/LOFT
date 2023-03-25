from envs.base_env import BaseEnv
from envs.cover import Cover
from envs.painting import Painting
from envs.blocks import Blocks
from envs.robosuite import Robosuite


def create_env(config):
    name_to_cls = {
        "cover": Cover,
        "painting": Painting,
        "blocks": Blocks,
        "robosuite": Robosuite,
    }
    return name_to_cls[config.env](config)
