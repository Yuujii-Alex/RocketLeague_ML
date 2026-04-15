from typing import List, Dict, Any, Tuple
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball"""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            player_vel = car.physics.linear_velocity
            pos_diff = state.ball.position - car.physics.position
            dist_to_ball = np.linalg.norm(pos_diff)

            if dist_to_ball < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_ball = pos_diff / dist_to_ball
            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            if speed_toward_ball > 0:
                rewards[agent] = float(speed_toward_ball / common_values.CAR_MAX_SPEED)
            else:
                rewards[agent] = 0.0
        return rewards


class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for facing the ball"""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            pos_diff = state.ball.position - car.physics.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_ball = pos_diff / dist
            dot = np.dot(car.physics.forward, dir_to_ball)
            rewards[agent] = float(max(dot, 0.0))
        return rewards
