from typing import SupportsFloat, Any
import typing as tp

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.core import ActType, ObsType, RenderFrame


class AvoidUnit:
    def __init__(self, initial_location: tp.Tuple[int, int], speed: int, seek_range: int):
        self.position: tp.Tuple[int, int] = initial_location
        self.speed: int = speed
        self.seek_range: int = seek_range

    def update(self, env: gym.Env):
        # choose a random direction to go in.
        chosen_action = env.action_space.sample()

        # We'll just trust that env is set up to be an AvoidEnvironment
        direction: tp.Tuple[int, int] = env.translate_action_to_direction(chosen_action)
        possible_position: np.ndarray = np.array(self.position) + np.array(direction) * self.speed

        self.position = tuple(np.clip(possible_position, 0, env.size - 1))

    def can_see_location(self, location: tp.Tuple[int, int]) -> bool:
        return abs(location[0] - self.position[0]) <= self.seek_range and abs(
            location[1] - self.position[1]) <= self.seek_range


class StationaryUnit(AvoidUnit):

    def __init__(self, location: tp.Tuple[int, int]):
        super().__init__(location, speed=0, seek_range=2)


class MobileUnit(AvoidUnit):

    def __init__(self, initial_location: tp.Tuple[int, int]):
        super().__init__(initial_location, speed=1, seek_range=1)


class AvoidEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 3}

    def __init__(self, num_mobile_units: int, num_stationary_units: int,
                 actor_seek_range: int, size: int = 13, render_mode: tp.Optional[str] = None, desired_steps: int = 75):
        super(AvoidEnvironment, self).__init__()

        self.num_mobile_units: int = num_mobile_units
        self.num_stationary_units: int = num_stationary_units

        self.opponent_units: tp.List[AvoidUnit] = []
        self.actor_seek_range: int = actor_seek_range
        self.size = size

        self.observation_space = gym.spaces.Dict({
            'agent': gym.spaces.Box(0, self.size, shape=(2,), dtype=int),
            'target': gym.spaces.Box(0, self.size, shape=(2,), dtype=int),
            'opponents': gym.spaces.Box(-1, self.size, shape=(self.num_stationary_units + self.num_mobile_units, 2),
                                        dtype=int)
        })

        # 5 Possible actions to take: Stay/Right/Up/Left/Down
        self.action_space = gym.spaces.Discrete(5)

        self._agent_location: tp.Optional[np.ndarray] = None
        self._target_location: tp.Optional[np.ndarray] = None
        self._original_distance: tp.Optional[int] = None

        self.num_steps: int = 0
        self.target_steps: int = desired_steps

        self._action_to_direction: tp.Dict[int, tp.Tuple[int, int]] = {
            # Action: X, Y
            0: (0, 0),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
            4: (0, -1)
        }

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.window_size = 720
        self.clock = None

        self.actor_seek_range: int = actor_seek_range

    def translate_action_to_direction(self, action: int) -> tp.Tuple[int, int]:
        return self._action_to_direction[action]

    def _get_observation(self) -> tp.Dict:
        output: tp.Dict = {
            'agent': self._agent_location,
            'target': self._target_location,
            'opponents': np.array(
                [unit.position if self._can_actor_see_unit(unit) else (-1, -1) for unit in self.opponent_units])
        }

        return output

    def _can_actor_see_unit(self, unit: AvoidUnit) -> bool:
        return abs(unit.position[0] - self._agent_location[0]) <= self.actor_seek_range and abs(
            unit.position[1] - self._agent_location[1]) <= self.actor_seek_range

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tp.Tuple[ObsType, dict[str, Any]]:
        super(AvoidEnvironment, self).reset(seed=seed)

        if options is not None and 'render_mode' in options.keys():
            render_mode: tp.Optional[str] = options['render_mode']
            assert render_mode is None or render_mode in self.metadata['render_modes']
            self.render_mode = render_mode

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        # If the agent can see the actor within half of its vision, the target location is too close.
        while (np.abs(self._agent_location - self._target_location) < self.actor_seek_range // 2).any():
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._original_distance: int = abs(self._agent_location[0] - self._target_location[0]) + abs(
            self._agent_location[1] - self._target_location[1])

        # Make all of the opponent units.
        self.opponent_units.clear()
        self.opponent_units.extend([MobileUnit(tuple(self.np_random.integers(0, self.size, size=2, dtype=int))) for i in
                                    range(self.num_mobile_units)])
        self.opponent_units.extend(
            [StationaryUnit(tuple(self.np_random.integers(0, self.size, size=2, dtype=int))) for i in
             range(self.num_stationary_units)])

        # Go over them again and make sure that the units can't already see the actor
        # nor are they stacked on top of each other.
        for unit in self.opponent_units:
            unit_can_see_agent: bool = unit.can_see_location(tuple(self._agent_location))
            unit_can_see_target: bool = unit.can_see_location(tuple(self._target_location))
            is_unit_on_unit: bool = any(
                [unit.position[0] == uunniitt.position[0] and unit.position[1] == uunniitt.position[1] for uunniitt in
                 self.opponent_units if uunniitt != unit])

            while unit_can_see_agent or is_unit_on_unit or unit_can_see_target:
                unit.position = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))

                # Update flags
                unit_can_see_agent: bool = unit.can_see_location(tuple(self._agent_location))
                unit_can_see_target: bool = unit.can_see_location(tuple(self._target_location))
                is_unit_on_unit: bool = any(
                    [unit.position[0] == uunniitt.position[0] and unit.position[1] == uunniitt.position[1] for uunniitt
                     in self.opponent_units if uunniitt != unit])

        observation: gym.spaces.Dict = self._get_observation()
        info: tp.Dict = {}
        self.num_steps = 0

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas: pygame.Surface = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size: float = self.window_size / self.size

        pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(pix_square_size * self._target_location,
                                                          (pix_square_size, pix_square_size)))
        pygame.draw.circle(canvas, (0, 255, 0), (self._agent_location + .5) * pix_square_size, pix_square_size / 3)

        for unit in self.opponent_units:
            pygame.draw.circle(canvas, (255, 0, 0), (np.array(unit.position) + .5) * pix_square_size,
                               pix_square_size / 3)

        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        direction: tp.Tuple[int, int] = self.translate_action_to_direction(action)

        self.num_steps += 1

        self._agent_location = np.clip(self._agent_location + np.array(direction), 0, self.size - 1)

        terminated: bool = np.array_equal(self._agent_location, self._target_location) or any(
            [unit.can_see_location(tuple(self._agent_location)) for unit in self.opponent_units])

        current_distance: int = abs(self._agent_location[0] - self._target_location[0]) + abs(
            self._agent_location[1] - self._target_location[1])

        distance_reward: float = (self._original_distance - current_distance) / self._original_distance
        units_in_sight_reward: float = len([1 for unit in self.opponent_units if self._can_actor_see_unit(unit)]) / len(
            self.opponent_units)
        agent_spotted_penalty: float = len(
            [1 for unit in self.opponent_units if unit.can_see_location(tuple(self._agent_location))]) / len(
            self.opponent_units)
        time_penalty: float = max(0, self.num_steps - self.target_steps) / self.target_steps

        distance_weight: float = .9
        units_in_sight_weight: float = .1
        agent_spotted_weight: float = -999999
        time_weight: float = -0.25

        reward: float = distance_weight * distance_reward + units_in_sight_weight * units_in_sight_reward + \
                        agent_spotted_weight * agent_spotted_penalty + time_weight * time_penalty

        # Move all the units in time before the next observation.
        for unit in self.opponent_units:
            unit.update(self)

        observation: gym.spaces.Dict = self._get_observation()
        info: tp.Dict[str, float] = {'distance_reward': distance_reward, 'units_in_sight_reward': units_in_sight_reward,
                                     'agent_spotted_penalty': -agent_spotted_penalty, 'time_penalty': -time_penalty}

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, False, info
