from collections import defaultdict
import gymnasium as gym
import numpy as np
import typing as tp



class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env: gym.Env = env
        self.q_values: tp.Dict = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr: float = learning_rate
        self.discount_factor: float = discount_factor

        self.epsilon: float = initial_epsilon
        self.epsilon_decay: float = epsilon_decay
        self.final_epsilon: float = final_epsilon

        self.training_error: tp.List[float] = []

    def get_action(self, obs: tp.Tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # Take random action to explore the action space of the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Given that epsilon will have made be smaller as we do more episodes,
        # we are likely to have found an optimal strategy by now.
        # Of course, that's not a guarantee, but we should be more experienced if this clause is met,
        # and therefore we're going to exploit that experience and take what we think is the optimal
        # action for the environment.
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tp.Tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tp.Tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        # Update the current experience we have with the environment by a small nudge towards what we think is the
        # optimal action.
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        # Epsilon is kinda like a measure of how inexperienced we are.
        # As we go through more episodes of exploring/exploitation, we tend to know more
        # about our environment and what actions are optimal, so we will be taking more exploitative
        # actions against our environment rather than choosing a random action and seeing how it
        # affects the environment.
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
