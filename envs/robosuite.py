"""Base class for an env.
"""

import numpy as np
import robosuite as suite
import structs
from structs import WORLD
from envs import BaseEnv

from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
import mujoco


class Robosuite(BaseEnv):
    """Robosuite environment.
    """
    def __init__(self, config):
        super().__init__(config)

        # Types
        self.block_type = structs.Type("block")
        self.pose_type = structs.ContinuousType("pose")
        self.pose_type.set_sampler(lambda rng: rng.uniform(-10, 10, size=3))

        # Predicates
        self.IsBlock = structs.Predicate(
            "IsBlock", 1, is_action_pred=False,
            holds=self._IsBlock_holds, var_types=[self.block_type])
        self.On = structs.Predicate(
            "On", 2, is_action_pred=False,
            holds=self._On_holds,
            var_types=[self.block_type, self.block_type])
        self.OnTable = structs.Predicate(
            "OnTable", 1, is_action_pred=False,
            holds=self._OnTable_holds, var_types=[self.block_type])
        self.Clear = structs.Predicate(
            "Clear", 1, is_action_pred=False,
            holds=self._Clear_holds, var_types=[self.block_type])
        self.Holding = structs.Predicate(
            "Holding", 1, is_action_pred=False,
            holds=self._Holding_holds, var_types=[self.block_type])
        self.HandEmpty = structs.Predicate(
            "HandEmpty", 0, is_action_pred=False,
            holds=self._HandEmpty_holds, var_types=[])

        # Action predicates
        self.Pick = structs.Predicate(
            "Pick", 1, is_action_pred=True,
            sampler=self._pick_sampler,
            var_types=[self.block_type])

    def get_state_predicates(self):
        """Set of state predicates in this environment.
        """
        return {self.IsBlock, self.On, self.OnTable, self.Clear, self.Holding,
                self.HandEmpty}

    def get_action_predicates(self):
        """Set of action predicates.
        """
        return {self.Pick}

    def _get_demo_problems(self, num):
        """Returns a list of planning problems for demo. Each is a tuple
        of (low-level initial state, goal). Goal is a LiteralConjunction that
        also implements a holds(state) method on low-level states.
        """
        problems = []
        goal1 = structs.LiteralConjunction(
            [self.Covers(self._blocks[0], self._targets[0])])
        goal2 = structs.LiteralConjunction(
            [self.Covers(self._blocks[1], self._targets[1])])
        goal3 = structs.LiteralConjunction(
            [self.Covers(self._blocks[0], self._targets[0]),
             self.Covers(self._blocks[1], self._targets[1])])
        goals = [goal1, goal2, goal3]
        for i in range(num):
            problems.append((self._create_initial_state(), goals[i % len(goals)]))
        return problems

    def get_test_problems(self):
        """Returns a list of planning problems for evaluation. Each is a tuple
        of (low-level initial state, goal). Goal is a LiteralConjunction that
        also implements a holds(state) method on low-level states.
        """
        raise NotImplementedError("Override me!")

    def get_next_state(self, state, action):
        """Transition function / simulator on low-level states/actions.
        Returns a next low-level state.
        """
        if action.predicate.var_types != [var.var_type for var
                                          in action.variables]:
            next_state = self._copy_state(state)
            return next_state

        # Pick
        if action.predicate == self.Pick:
            return self._get_next_state_pick(state, action)

    def _get_next_state_pick(self, state, action):
        assert action.predicate == self.Pick
        next_state = self._copy_state(state)
        # Can only pick if hand is empty
        if state[WORLD]["cur_holding"] is not None:
            return next_state
        # Execute pick
        next_state[WORLD]["cur_holding"] = obj
        next_state[WORLD]["block_to_pile_idx"][obj] = None
        next_state[WORLD]["piles"][pile_idx].pop(-1)
        next_state[WORLD]["cur_grip"] = state[obj]["pose"]
        next_state[obj]["held"] = True
        return next_state

    def get_random_action(self, state):
        """Get a random valid action from the given state.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _IsBlock_holds(state, obj):
        return obj in state and obj.var_type == "block"

    @staticmethod
    def _On_holds(state, obj1, obj2):
        pile_idx1 = state[WORLD]['block_to_pile_idx'][obj1]
        pile_idx2 = state[WORLD]['block_to_pile_idx'][obj2]
        # One of the blocks is held
        if pile_idx1 is None or pile_idx2 is None:
            return False
        return (pile_idx1[0] == pile_idx2[0] and
                pile_idx1[1] == pile_idx2[1] + 1)

    @staticmethod
    def _Holding_holds(state, obj):
        holding = (state[WORLD]["cur_holding"] is not None and
                   state[WORLD]["cur_holding"] == obj)
        assert holding == (state[WORLD]['block_to_pile_idx'][obj] \
                           is None)
        assert holding == state[obj]["held"]
        return holding

    @staticmethod
    def _OnTable_holds(state, obj):
        pile_idx = state[WORLD]['block_to_pile_idx'][obj]
        return pile_idx is not None and pile_idx[1] == 0

    @staticmethod
    def _Clear_holds(state, obj):
        pile_idx = state[WORLD]['block_to_pile_idx'][obj]
        if pile_idx is None:
            return False
        pile_size = len(state[WORLD]['piles'][pile_idx[0]])
        return pile_idx[1] == pile_size-1

    @staticmethod
    def _HandEmpty_holds(state):
        return state[WORLD]["cur_holding"] is None

    @staticmethod
    def _pick_sampler(rng, state, *args):
        del rng  # unused
        del state  # unused
        del args  # unused
        return tuple()

    @staticmethod
    def _initial_robosuite_setup(self):
        """One-time robosuite setup stuff.
        """
        # create environment instance
        env = suite.make(
            env_name="Stack",
            robots="Jaco",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )

        env.reset()

        for i in range(1000):
            action = np.random.randn(env.robots[0].dof)  # sample random action
            obs, reward, done, info = env.step(action)  # take action in the environment
            env.render()  # render on display