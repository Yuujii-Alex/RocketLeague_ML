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


class DribbleDecayReward(RewardFunction[AgentID, GameState, float]):
    """
    Stateful reward that prevents touch-farming.
    The reward multiplier (lambda) decays on every touch and recharges when not touching.
    Based on the "Seer" Dribble Penalty logic.
    """

    def __init__(self):
        super().__init__()
        self.lambdas = {}

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        for agent in agents:
            self.lambdas[agent] = 1.0

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
            # Ensure agent exists in our state
            if agent not in self.lambdas:
                self.lambdas[agent] = 1.0

            # Check if agent touched the ball
            if state.cars[agent].ball_touches > 0:
                # Give reward scaled by current lambda
                rewards[agent] = float(self.lambdas[agent])
                # Decay lambda (shrink the prize for the next touch)
                # 0.95 multiplier means ~13 continuous touches reaches the floor
                self.lambdas[agent] = max(0.1, self.lambdas[agent] * 0.95)
            else:
                rewards[agent] = 0.0
                # Recharge lambda (restore the prize over time)
                # 0.013 addition means full recharge takes ~70 ticks (~4.5s)
                self.lambdas[agent] = min(1.0, self.lambdas[agent] + 0.013)

        return rewards


class BallDistanceReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being close to the ball using exponential decay."""

    def __init__(self, distance_scale: float = 1000.0):
        super().__init__()
        self.distance_scale = distance_scale

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
            pos_diff = state.ball.position - state.cars[agent].physics.position
            dist = np.linalg.norm(pos_diff)
            # Exponential decay: e^(-dist / scale)
            reward = np.exp(-dist / self.distance_scale)
            rewards[agent] = float(reward)
        return rewards


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal"""

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
            ball = state.ball

            # Blue goal at -5120, Orange goal at 5120
            if car.team_num == common_values.ORANGE_TEAM:
                goal_y = -common_values.BACK_WALL_Y
            else:
                goal_y = common_values.BACK_WALL_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_goal = pos_diff / dist
            vel_toward_goal = np.dot(ball_vel, dir_to_goal)

            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0.0)
        return rewards


class AlignBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being positioned behind the ball relative to the goal"""

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
            ball = state.ball

            # Opponent goal location
            if car.team_num == common_values.ORANGE_TEAM:
                goal_y = -common_values.BACK_WALL_Y
            else:
                goal_y = common_values.BACK_WALL_Y

            goal_pos = np.array([0, goal_y, 0])

            # Vector from goal to ball
            ball_to_goal = goal_pos - ball.position
            ball_to_goal /= np.linalg.norm(ball_to_goal)

            # Vector from player to ball
            player_to_ball = ball.position - car.physics.position
            player_to_ball /= np.linalg.norm(player_to_ball)

            # Dot product: 1.0 means player is perfectly behind the ball relative to goal
            alignment = np.dot(ball_to_goal, player_to_ball)

            # Squared drop-off
            rewards[agent] = float(max(alignment, 0.0) ** 2)
        return rewards


class PowerHitReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball hard (based on change in ball velocity)"""

    def __init__(self):
        super().__init__()
        self.last_ball_vel = None

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}

        current_ball_vel = state.ball.linear_velocity
        if self.last_ball_vel is not None:
            # Calculate the magnitude of the velocity change
            delta_v = np.linalg.norm(current_ball_vel - self.last_ball_vel)

            # Check for touches
            for agent in agents:
                if state.cars[agent].ball_touches > 0:
                    # Reward based on how much the ball velocity changed
                    # Normalizing by BALL_MAX_SPEED (6000)
                    rewards[agent] = float(delta_v / common_values.BALL_MAX_SPEED)

        self.last_ball_vel = current_ball_vel.copy()
        return rewards


class SaveBoostReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for having boost. Uses sqrt to prioritize low-end boost management."""

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
        # Reward is the square root of boost amount (0 to 1 range)
        return {
            agent: float(np.sqrt(state.cars[agent].boost_amount / 100.0))
            for agent in agents
        }


class AggressiveGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards for scoring (+10) and punishes the team for conceding (-8.0) to promote aggression."""

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
        rewards = {agent: 0.0 for agent in agents}
        if state.goal_scored:
            for agent in agents:
                if state.scoring_team == state.cars[agent].team_num:
                    rewards[agent] = 10.0
                else:
                    rewards[agent] = -8.0
        return rewards


class DemoReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for demolishing an opponent"""

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
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            if car.bump_victim_id is not None:
                victim = state.cars.get(car.bump_victim_id)
                if victim and victim.is_demoed:
                    # Reward the attacker
                    rewards[agent] += 1.0
                    # Punish the victim (Zero-Sum)
                    rewards[car.bump_victim_id] -= 1.0
        return rewards


class ConstantStepReward(RewardFunction[AgentID, GameState, float]):
    """Returns a constant reward of 1.0 on every step. Useful for step penalties."""

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
        return {agent: 1.0 for agent in agents}


class AerialGoalStrikeReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the goal specifically while in the air. (min_height = 200)"""

    def __init__(self, min_height: float = 200.0):
        super().__init__()
        self.last_ball_vel = None
        self.min_height = min_height

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        current_ball_vel = state.ball.linear_velocity

        if self.last_ball_vel is not None:
            # Check for touches while in the air AND ball is above min_height
            for agent in agents:
                car = state.cars[agent]
                if (
                    car.ball_touches > 0
                    and not car.on_ground
                    and state.ball.position[2] > self.min_height
                ):
                    # Get goal direction
                    if car.team_num == common_values.ORANGE_TEAM:
                        goal_y = -common_values.BACK_WALL_Y
                    else:
                        goal_y = common_values.BACK_WALL_Y

                    goal_pos = np.array([0.0, goal_y, 0.0])
                    pos_diff = goal_pos - state.ball.position
                    dist = np.linalg.norm(pos_diff)

                    if dist > 1e-6:
                        dir_to_goal = pos_diff / dist

                        # Calculate velocity toward goal before and after
                        prev_vel_to_goal = np.dot(self.last_ball_vel, dir_to_goal)
                        curr_vel_to_goal = np.dot(current_ball_vel, dir_to_goal)

                        # Reward the change (improvement) in goal-ward velocity
                        delta_v_to_goal = curr_vel_to_goal - prev_vel_to_goal
                        if delta_v_to_goal > 0:
                            # Using 15.0 multiplier in Shen.py, so we keep the base small here
                            rewards[agent] = float(
                                delta_v_to_goal / common_values.BALL_MAX_SPEED
                            )

        self.last_ball_vel = current_ball_vel.copy()
        return rewards


class FirstTouchReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being the first to touch the ball, but only if it goes toward the opponent's half."""

    def __init__(self):
        super().__init__()
        self.first_touch_occurred = False

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.first_touch_occurred = False

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}

        if not self.first_touch_occurred:
            for agent in agents:
                car = state.cars[agent]
                if car.ball_touches > 0:
                    # A touch occurred! We mark it as occurred so nobody else can get it this kickoff.
                    self.first_touch_occurred = True

                    # Directional check: is the ball now moving toward the opponent's goal?
                    # Blue (0) attacks Positive Y, Orange (1) attacks Negative Y
                    ball_vel_y = state.ball.linear_velocity[1]
                    is_forward = (car.team_num == common_values.BLUE_TEAM and ball_vel_y > 50) or \
                                 (car.team_num == common_values.ORANGE_TEAM and ball_vel_y < -50)

                    if is_forward:
                        # Reward only if it was a productive first touch
                        rewards[agent] = 1.0
                    break

        return rewards


class LandOnWheelsReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for landing on its wheels after being in the air, with height check to prevent farming."""

    def __init__(self, min_height: float = 250.0):
        super().__init__()
        self.last_on_ground = {}
        self.max_height_attained = {}
        self.min_height = min_height

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.last_on_ground = {agent: True for agent in agents}
        self.max_height_attained = {agent: 0.0 for agent in agents}

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            current_on_ground = car.on_ground
            last_on_ground = self.last_on_ground.get(agent, True)

            if not current_on_ground:
                # Track max height in flight
                self.max_height_attained[agent] = max(
                    self.max_height_attained.get(agent, 0.0), car.physics.position[2]
                )

            # Trigger only at the moment of landing
            if not last_on_ground and current_on_ground:
                # Only reward if they actually jumped/flew high enough
                if self.max_height_attained.get(agent, 0.0) > self.min_height:
                    up_vector = car.physics.up
                    alignment = up_vector[2]  # World Z is [0, 0, 1]

                    # Reward if upright, penalize if on back/side
                    rewards[agent] = float(alignment)

                # Reset height tracking for next flight
                self.max_height_attained[agent] = 0.0

            self.last_on_ground[agent] = current_on_ground

        return rewards


class BoostTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for using boost while facing and moving toward the ball."""

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
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            # Check if boosting
            if car.is_boosting:
                # Calculate velocity toward ball
                pos_diff = state.ball.position - car.physics.position
                dist = np.linalg.norm(pos_diff)
                if dist > 1e-6:
                    dir_to_ball = pos_diff / dist
                    speed_toward_ball = np.dot(car.physics.linear_velocity, dir_to_ball)

                    # Reward only if moving toward ball while boosting
                    if speed_toward_ball > 0:
                        # Scaling: 1.0 reward at max car speed
                        rewards[agent] = float(
                            speed_toward_ball / common_values.CAR_MAX_SPEED
                        )
        return rewards


class AerialStabilityReward(RewardFunction[AgentID, GameState, float]):
    """Nudges the agent to learn flight by rewarding boosting while pointing up in the air."""

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
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            car = state.cars[agent]
            # In air AND boosting AND nose is pointed up
            if (
                not car.on_ground
                and car.is_boosting
                and car.physics.forward[2] > 0.3
            ):
                # Tiny constant reward to act as a 'signal'
                rewards[agent] = 0.1
        return rewards


class SupersonicReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in a supersonic state."""

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(
        self, agents, state, is_terminated, is_truncated, shared_info
    ) -> Dict[AgentID, float]:
        return {agent: float(state.cars[agent].is_supersonic) for agent in agents}


class ConserveBoostReward(RewardFunction[AgentID, GameState, float]):
    """Very light penalty for using boost to discourage absolute waste."""

    def __init__(self):
        super().__init__()
        self.last_boost = {}

    def reset(self, agents, initial_state, shared_info) -> None:
        self.last_boost = {
            agent: initial_state.cars[agent].boost_amount for agent in agents
        }

    def get_rewards(
        self, agents, state, is_terminated, is_truncated, shared_info
    ) -> Dict[AgentID, float]:
        rewards = {agent: 0.0 for agent in agents}
        for agent in agents:
            current_boost = state.cars[agent].boost_amount
            last_boost = self.last_boost.get(agent, current_boost)

            if last_boost > current_boost:
                delta = last_boost - current_boost
                rewards[agent] = -float(delta / 100.0)

            self.last_boost[agent] = current_boost
        return rewards
