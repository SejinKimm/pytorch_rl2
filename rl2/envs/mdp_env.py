"""
Implements the Tabular MDP environment(s) from Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple
import torch
import operator
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv
from torch.utils.data import Dataset

ACTION_DIC={
    "start":0,
    "undo":1,
    "edit":2,
    "copyFromInput":3,
    "rotate":4,
    "reflecty":5,
    "reflectx":6,
    "translate":7,
    "resizeOutputGrid":8,
    "resetOutputGrid":9,
    "selected_cells":10, 
    "select_fill":11,
    "none":12,
    "end":13,
}
    
class Train_StateActionReturnDataset(Dataset):
    def __init__(self, data_path, data_type=None, step_gap=5, block_size=5*5+2): 
        self.actions_type = ACTION_DIC
        self.vocab_size = len(self.actions_type)
        self.block_size = block_size
        self.step_gap = step_gap
        self.data_path = Path(data_path)
        self.training_path = self.data_path / data_type
        self.training_tasks = []


        for path, _, files in os.walk(self.training_path):
            for name in files:
                self.training_tasks.append(os.path.join(path, name))

        self.num_dataset = len(self.training_tasks)
        
    def __len__(self):
        return self.num_dataset

    def __getitem__(self, idx):
        task_file = self.training_tasks[idx]
            
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        #total_step = int(task["total_step"])

        # start_idx = np.random.randint(-self.step_gap+2, total_step-self.step_gap+1)
        
        pnp = []
        states = []
        actions_str = []
        actions = []
        rtgs = []
        timesteps = []
        
        for i in range(len(task["action_sequence"]["action_sequence"])):
            pnp.append(reduce(operator.add, task["action_sequence"]["action_sequence"][i]["pnp"]))
            states.append(reduce(operator.add, task["action_sequence"]["action_sequence"][i]["grid"]))
            actions_str.append(task["action_sequence"]["action_sequence"][i]["action"]["tool"])
            actions.append(self.actions_type[task["action_sequence"]["action_sequence"][i]["action"]["tool"]])
            rtgs.append(float(task["action_sequence"]["action_sequence"][i]["reward"]))
            timesteps.append(int(task["action_sequence"]["action_sequence"][i]["time_step"]))
        
        
        pnp = torch.LongTensor(pnp)
        #print(pnp.shape)
        states = torch.LongTensor(states)
        states = states.view([-1])
        states = states.contiguous()
        actions = torch.LongTensor(actions)
        rtgs = torch.FloatTensor(rtgs)
        timesteps = torch.LongTensor(timesteps)
        
        return states, actions, rtgs, timesteps, pnp

train_dataset = Train_StateActionReturnDataset(
    data_path = 'ARC_Data',
    data_type = 'mini_arc_train', 
    step_gap = 6
    block_size = 6
)

class MDPEnv(MetaEpisodicEnv):
    """
    Tabular MDP env with support for resettable MDP params (new meta-episode),
    in addition to the usual reset (new episode).
    """
    def __init__(self, num_states, num_actions, max_episode_length=10):
        # structural
        self._num_states = num_states
        self._num_actions = num_actions
        self._max_ep_length = max_episode_length

        # per-environment-sample quantities.
        self._reward_means = None
        self._state_transition_probabilities = None
        self.new_env()

        # mdp state.
        self._ep_steps_so_far = 0
        self._state = 0

    @property
    def max_episode_len(self):
        return self._max_ep_length

    @property
    def num_actions(self):
        """Get self._num_actions."""
        return self._num_actions

    @property
    def num_states(self):
        """Get self._num_states."""
        return self._num_states

    def _new_reward_means(self):
        self._reward_means = np.random.normal(
            loc=1.0, scale=1.0, size=(self._num_states, self._num_actions))

    def _new_state_transition_dynamics(self):
        p_aijs = []
        for a in range(self._num_actions):
            dirichlet_samples_ij = np.random.dirichlet(
                alpha=np.ones(dtype=np.float32, shape=(self._num_states,)),
                size=(self._num_states,))
            p_aijs.append(dirichlet_samples_ij)
        self._state_transition_probabilities = np.stack(p_aijs, axis=0)

    def new_env(self) -> None:
        """
        Sample a new MDP from the distribution over MDPs.

        Returns:
            None
        """
        self._new_reward_means()
        self._new_state_transition_dynamics()
        self._state = 0

    def reset(self) -> int:
        """
        Reset the environment.

        Returns:
            initial state.
        """
        self._ep_steps_so_far = 0
        self._state = 0
        return self._state

    def step(self, action, auto_reset=True) -> Tuple[int, float, bool, dict]:
        """
        Take action in the MDP, and observe next state, reward, done, etc.

        Args:
            action: action corresponding to an arm index.
            auto_reset: auto reset. if true, new_state will be from self.reset()

        Returns:
            new_state, reward, done, info.
        """
        self._ep_steps_so_far += 1
        t = self._ep_steps_so_far

        s_t = self._state
        a_t = action

        s_tp1 = np.random.choice(
            a=self._num_states,
            p=self._state_transition_probabilities[a_t, s_t])
        self._state = s_tp1

        r_t = np.random.normal(
            loc=self._reward_means[s_t, a_t],
            scale=1.0)

        done_t = False if t < self._max_ep_length else True
        if done_t and auto_reset:
            s_tp1 = self.reset()

        return s_tp1, r_t, done_t, {}
