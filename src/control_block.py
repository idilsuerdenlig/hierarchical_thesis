from block import Block
from J_block import JBlock
from M_block import MBlock
from mushroom.algorithms import agent

class ControlBlock(Block):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, wake_time, agent, episode_length, fit_time, n_iterations=1):
        """
        Constructor.

        Args:
            policy (object): the policy to use for the agent;
            gamma (float): discount factor;
            params (dict): other parameters of the algorithm.
            n_iterations: number of iterations for the fit of the agent
        """

        self.agent = agent
        self.step_counter = 0
        self.episode_counter = 0
        self.dataset = list()
        self.n_iterations = n_iterations
        self.episode_length = episode_length
        self.fit_time = fit_time
        self.last_input = None
        self.last_reward = None
        self.last_output = None
        self.last_abs = False
        self.last_last = False
        self.fit_flag = False

        super(ControlBlock, self).__init__(wake_time=wake_time)


    def __call__(self, inputs, reward, absorbing, learn_flag):
        """
        Draw action: Returns the action to execute. It is the action returned by the policy
        or the action set by the algorithm (e.g. SARSA).

        Args:
            r state (np.array): the state where the agent is.
            absorbing: state bein absorbing for the subtask or for the mdp
            wake: wake_time of the block
            fit: fit time of the block

        Returns:
            The action to be executed.

        """
        self.clock_counter+=1

        if learn_flag and self.episode_counter == self.fit_time:
            self.fit(self.dataset, self.n_iterations)

        if absorbing or self.step_counter == self.episode_length :
            self.agent.episode_start()
            self.dataset.append((self.last_input, self.last_output, self.last_reward, inputs, True))
            self.last_output = None
            self.last_reward = None
            self.last_input = None
            self.last_abs = False
            self.last_last = False
            self.clock_counter = 0
            self.episode_counter+=1

        elif self.wake_time == self.clock_counter:
             print'control block draw action'
             action = self.agent.draw_action(state=inputs)
             self.clock_counter=0
             if not self.fit_flag:
                 self.dataset.append((self.last_input, self.last_output, self.last_reward,
                                      inputs , self.last_last))
             else:
                 self.fit_flag = False
             self.last_output = action
             self.last_input = inputs
             self.last_reward = reward
             self.last_abs = absorbing
             self.step_counter+=1
             self.last_last = self.last_abs or self.step_counter == self.episode_length

        return absorbing

    def fit(self, dataset, n_iterations):

        self.agent.fit(dataset, n_iterations)
        self.episode_counter=0


    def add_reward(self, reward_block):

        assert isinstance(reward_block, JBlock) or isinstance(reward_block, MBlock)
        self.reward_connection = reward_block
