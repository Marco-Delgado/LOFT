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
        self.PutOnTable = structs.Predicate(
            "PutOnTable", 1, is_action_pred=True,
            sampler=self._put_on_table_sampler,
            var_types=[self.pose_type])

        self._initial_robosuite_setup()

    def get_state_predicates(self):
        """Set of state predicates in this environment.
        """
        return {self.IsBlock, self.On, self.OnTable, self.Clear, self.Holding,
                self.HandEmpty}

    def get_action_predicates(self):
        """Set of action predicates.
        """
        return {self.Pick, self.PutOnTable}

    def _get_demo_problems(self, num):
        """Returns a list of planning problems for demo. Each is a tuple
        of (low-level initial state, goal). Goal is a LiteralConjunction that
        also implements a holds(state) method on low-level states.
        """
        return self._get_problems(
            num, self._cf.blocks_demo_num_objs)

    def get_test_problems(self):
        """Returns a list of planning problems for evaluation. Each is a tuple
        of (low-level initial state, goal). Goal is a LiteralConjunction that
        also implements a holds(state) method on low-level states.
        """
        return self._get_problems(
            self._cf.blocks_num_test_problems,
            self._cf.blocks_test_num_objs)

    def _get_problems(self, num_problems, all_num_objs):
        problems = []
        for i in range(num_problems):
            j = i % len(all_num_objs)
            num_objs = all_num_objs[j]
            problems.append(self._create_problem(num_objs))
        return problems

    def _create_problem(self, num_objs):
        # Sample piles
        piles = self._sample_initial_piles(num_objs)
        # Create state from piles
        state = self._sample_state_from_piles(piles)
        while True:  # repeat until goal is not satisfied
            # Sample goal
            goal = self._sample_goal_from_piles(num_objs, piles)
            if not goal.holds(state):
                break
        return state, goal

    def _sample_initial_piles(self, num_objs):
        piles = []
        for block_num in range(num_objs):
            block = self.block_type(f"block{block_num}")
            # If coin flip, start new pile
            if block_num == 0 or self._rng.uniform() < 0.2:
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles

    def _sample_goal_from_piles(self, num_objs, piles):
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_objs)
            if goal_piles != piles:
                break
        # Create literal goal from piles
        goal_lits = []
        for pile in goal_piles:
            goal_lits.append(self.OnTable(pile[0]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_lits.append(self.On(block1, block2))
        return structs.LiteralConjunction(goal_lits)

    def _sample_state_from_piles(self, piles):
        state = {}
        # Create world state
        world_state = {}
        world_state["cur_grip"] = [0.9, 0.3, 0.3]
        world_state["cur_holding"] = None
        world_state["piles"] = piles
        block_to_pile_idx = \
            self._create_block_to_pile_idx(piles)
        world_state["block_to_pile_idx"] = block_to_pile_idx
        state[WORLD] = world_state
        # Sample pile (x, y)s
        pile_to_xy = {}
        for i in range(len(piles)):
            (x, y) = self._sample_initial_pile_xy(
                self._rng, pile_to_xy.values())
            pile_to_xy[i] = (x, y)
        # Create block states
        for block, pile_idx in block_to_pile_idx.items():
            x, y = pile_to_xy[pile_idx[0]]
            z = self.table_height + self.block_size * (0.5 + pile_idx[1])
            block_state = {"pose": [x, y, z], "held": False}
            state[block] = block_state
        # Update world flat features
        self._update_world_flat(state)
        return state

    @staticmethod
    def _create_block_to_pile_idx(piles):
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        return block_to_pile_idx

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
        # PutOnTable
        if action.predicate == self.PutOnTable:
            return self._get_next_state_put_on_table(state, action)
        # Stack
        if action.predicate == self.Stack:
            return self._get_next_state_stack(state, action)
        raise Exception(f"Unexpected action: {action}")

    def _get_next_state_pick(self, state, action):
        assert action.predicate == self.Pick
        next_state = self._copy_state(state)
        # Can only pick if hand is empty
        if state[WORLD]["cur_holding"] is not None:
            return next_state
        # Can only pick if object is at top of pile
        obj = action.variables[0]
        pile_idx, in_pile_idx = state[WORLD]["block_to_pile_idx"][obj]
        if in_pile_idx != len(state[WORLD]["piles"][pile_idx]) - 1:
            return next_state
        # Execute pick
        next_state[WORLD]["cur_holding"] = obj
        next_state[WORLD]["block_to_pile_idx"][obj] = None
        next_state[WORLD]["piles"][pile_idx].pop(-1)
        next_state[WORLD]["cur_grip"] = state[obj]["pose"]
        next_state[obj]["held"] = True
        # Update world flat features
        self._update_world_flat(next_state)
        return next_state

    def _get_next_state_put_on_table(self, state, action):
        assert action.predicate == self.PutOnTable
        next_state = self._copy_state(state)
        # Can only put on table if holding
        holding_obj = next_state[WORLD]["cur_holding"]
        if holding_obj is None:
            return next_state
        # Can only put on table if pose is clear
        x, y, _ = action.variables[0].value
        obj_poses = [state[o]['pose'] for o in state if o != WORLD]
        existing_xys = [(p[0], p[1]) for p in obj_poses]
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute put on table
        pile_idx = len(state[WORLD]["piles"])
        new_pose = [x, y, self.table_height + 0.5 * self.block_size]
        next_state[WORLD]["cur_holding"] = None
        next_state[WORLD]["block_to_pile_idx"][holding_obj] = \
            (pile_idx, 0)
        next_state[WORLD]["piles"].append([holding_obj])
        next_state[WORLD]["cur_grip"] = new_pose
        next_state[holding_obj]["pose"] = new_pose
        next_state[holding_obj]["held"] = False
        # Update world flat features
        self._update_world_flat(next_state)
        return next_state

    def _get_next_state_stack(self, state, action):
        assert action.predicate == self.Stack
        next_state = self._copy_state(state)
        # Can only stack if holding
        holding_obj = next_state[WORLD]["cur_holding"]
        if holding_obj is None:
            return next_state
        # Can't stack on the object that we're holding
        obj = action.variables[0]
        if holding_obj == obj:
            return next_state
        # Can only stack if target is clear
        pile_idx, in_pile_idx = state[WORLD]["block_to_pile_idx"][obj]
        if in_pile_idx != len(state[WORLD]["piles"][pile_idx]) - 1:
            return next_state
        # Execute stack
        x, y, z = state[obj]["pose"]
        new_pose = [x, y, z + self.block_size]
        next_state[WORLD]["cur_holding"] = None
        next_state[WORLD]["block_to_pile_idx"][holding_obj] = \
            (pile_idx, in_pile_idx + 1)
        next_state[WORLD]["piles"][pile_idx].append(holding_obj)
        next_state[WORLD]["cur_grip"] = new_pose
        next_state[holding_obj]["pose"] = new_pose
        next_state[holding_obj]["held"] = False
        # Update world flat features
        self._update_world_flat(next_state)
        return next_state

    def get_random_action(self, state):
        """Get a random valid action from the given state.
        """
        objs = list(sorted(state.keys()))
        objs.remove(WORLD)
        act_pred = self._rng.randint(2)
        if act_pred == 0:  # Pick
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Pick, [obj])
        if act_pred == 1:  # Put on table
            return self._sample_ground_act(state, self.PutOnTable, [])
        raise Exception("Can never reach here")

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
        holding = (state[WORLD]["cur_holding"] is not None and state[WORLD]["cur_holding"] == obj)
        assert holding == (state[WORLD]['block_to_pile_idx'][obj] is None)
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
        return pile_idx[1] == pile_size - 1

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
    def _put_on_table_sampler(cls, rng, state, *args):
        del state  # unused
        del args  # unused
        (x, y) = Robosuite._sample_xy(rng)
        pose = [x, y, Robosuite.table_height + 0.5 * Robosuite.block_size]
        return (pose,)

    @staticmethod
    def _initial_robosuite_setup():
        """One-time robosuite setup stuff.
        """
        # create environment instance
        # env = suite.make(
        #     env_name="Stack",
        #     robots="Jaco",
        #     has_renderer=True,
        #     has_offscreen_renderer=False,
        #     use_camera_obs=False,
        # )
        #
        # env.reset()

    block_size = 0.06
    table_height = 0.2
    x_lb = 1.3
    x_ub = 1.4
    y_lb = 0.15
    y_ub = 0.85

    @staticmethod
    def _update_world_flat(state):
        """Set flat world features
        """
        holding_something = state[WORLD]["cur_holding"] is not None
        flat_feats = np.array([int(holding_something)])
        state[WORLD]["flat"] = flat_feats
        state[WORLD]["flat_names"] = np.array(["flat:holding_something"])

    @staticmethod
    def _sample_xy(rng):
        x = rng.uniform(Robosuite.x_lb, Robosuite.x_ub)
        y = rng.uniform(Robosuite.y_lb, Robosuite.y_ub)
        return (x, y)