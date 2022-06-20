from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
# Dependency imports
from absl import app
from absl import flags

from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui

import numpy as np
from pycolab import rendering

demo = demonstrations.get_demonstrations('rocks_diamonds')[0]
np.random.seed(demo.seed)
env = factory.get_environment_obj(environment_name)
env.reset()
episode_return = 0
for action in demo.actions:
    timestep = env.step(action)
    episode_return += timestep.reward if timestep.reward else 0
assert episode_return == demo.episode_return
