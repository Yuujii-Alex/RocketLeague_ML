import numpy as np
from typing import Dict, List, Any
from rlgym.api import RewardFunction
from rlgym.rocket_league.api import GameState

class TouchBallReward(RewardFunction):
    def __init__(self, min_time_between_touches: float = 1.5):
        super().__init__()
        self.min_time_between_touches = min_time_between_touches
        self.last_touch_times = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_touch_times = {agent: -self.min_time_between_touches for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        current_time = state.tick_count / 120.0
        
        for agent_id in agents:
            car = state.cars[agent_id]
            reward = 0.0
            last_touch_time = self.last_touch_times.get(agent_id, -self.min_time_between_touches)
            
            if car.ball_touches > 0 and (current_time - last_touch_time) >= self.min_time_between_touches:
                reward = 1.0
                self.last_touch_times[agent_id] = current_time
                
            rewards[agent_id] = float(reward)
            
            # Debug NaNs
            if np.isnan(rewards[agent_id]):
                print(f"NaN from TouchBallReward! agent={agent_id}")
                
        return rewards

class SpeedTowardBallReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        
        for agent_id in agents:
            car = state.cars[agent_id]
            car_pos = car.physics.position
            ball_pos = state.ball.position
            
            pos_diff = ball_pos - car_pos
            dist = float(np.linalg.norm(pos_diff))
            
            if dist < 1e-5:
                rewards[agent_id] = 0.0
                continue
                
            dir_to_ball = pos_diff / dist
            vel = car.physics.linear_velocity
            speed_toward_ball = np.dot(vel, dir_to_ball)
            
            rewards[agent_id] = float(speed_toward_ball / 2300.0)
            
            if np.isnan(rewards[agent_id]):
                print(f"NaN from SpeedTowardBallReward! dist={dist} vel={vel} dir={dir_to_ball}")
            
        return rewards

class FaceBallReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        
        for agent_id in agents:
            car = state.cars[agent_id]
            car_pos = car.physics.position
            ball_pos = state.ball.position
            
            pos_diff = ball_pos - car_pos
            dist = float(np.linalg.norm(pos_diff))
            
            if dist < 1e-5:
                rewards[agent_id] = 0.0
                continue
                
            dir_to_ball = pos_diff / dist
            forward = car.physics.forward
            alignment = np.dot(forward, dir_to_ball)
            
            rewards[agent_id] = float(alignment)
            
            if np.isnan(rewards[agent_id]):
                print(f"NaN from FaceBallReward! forward={forward}, dir_to_ball={dir_to_ball}")
            
        return rewards

class AerialTouchReward(RewardFunction):
    def __init__(self, min_height: float = 300.0, min_time_between_touches: float = 1.5):
        super().__init__()
        self.min_height = min_height
        self.min_time_between_touches = min_time_between_touches
        self.last_touch_times = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_touch_times = {agent: -self.min_time_between_touches for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        current_time = state.tick_count / 120.0
        
        for agent_id in agents:
            car = state.cars[agent_id]
            reward = 0.0
            last_touch_time = self.last_touch_times.get(agent_id, -self.min_time_between_touches)
            
            if car.ball_touches > 0 and car.physics.position[2] > self.min_height and (current_time - last_touch_time) >= self.min_time_between_touches:
                reward = 1.0
                self.last_touch_times[agent_id] = current_time
                
            rewards[agent_id] = float(reward)
            
            if np.isnan(rewards[agent_id]):
                print(f"NaN from AerialTouchReward! height={car.physics.position[2]}")
            
        return rewards

from rlgym.rocket_league import common_values

class DistanceBallGoalReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        # Seer's scaling factor: pos_net_y - pos_backwall_y + rball -> approx 10332.75
        c = 10332.75 
        
        for agent_id in agents:
            car = state.cars[agent_id]
            target_y = 5120.0 if car.team_num == 0 else -5120.0
            target_net = np.array([0.0, target_y, 0.0], dtype=np.float32)

            ball_pos = state.ball.position
            dist = float(np.linalg.norm(ball_pos - target_net))

            rewards[agent_id] = float(np.exp(-0.5 * dist / c))

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            target_y = 5120 if car.team_num == 0 else -5120
            target_net = np.array([0, target_y, 0], dtype=np.float32)
            
            ball_pos = state.ball.position
            pos_diff = target_net - ball_pos
            dist = float(np.linalg.norm(pos_diff))
            
            if dist < 1e-5:
                rewards[agent_id] = 0.0
                continue
                
            dir_to_goal = pos_diff / dist
            ball_vel = state.ball.linear_velocity
            speed_toward_goal = np.dot(ball_vel, dir_to_goal)
            
            rewards[agent_id] = float(speed_toward_goal / common_values.BALL_MAX_SPEED)
            
        return rewards

class AlignBallGoalReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            car_pos = car.physics.position
            ball_pos = state.ball.position

            # Determine the nets based on team
            # Blue (0) defends negative Y (-5120), attacks positive Y (5120)
            # Orange (1) defends positive Y (5120), attacks negative Y (-5120)
            if car.team_num == 0:
                pos_net_self = np.array([0.0, -5120.0, 0.0], dtype=np.float32)
                pos_net_opp = np.array([0.0, 5120.0, 0.0], dtype=np.float32)
            else:
                pos_net_self = np.array([0.0, 5120.0, 0.0], dtype=np.float32)
                pos_net_opp = np.array([0.0, -5120.0, 0.0], dtype=np.float32)
            
            vec_ball_car = ball_pos - car_pos
            vec_car_net_self = car_pos - pos_net_self
            vec_car_ball = car_pos - ball_pos
            vec_net_opp_car = pos_net_opp - car_pos

            norm_ball_car = float(np.linalg.norm(vec_ball_car))
            norm_car_net_self = float(np.linalg.norm(vec_car_net_self))
            norm_car_ball = float(np.linalg.norm(vec_car_ball))
            norm_net_opp_car = float(np.linalg.norm(vec_net_opp_car))

            # Avoid division by zero
            if (norm_ball_car < 1e-5 or norm_car_net_self < 1e-5 or 
                norm_car_ball < 1e-5 or norm_net_opp_car < 1e-5):
                rewards[agent_id] = 0.0
                continue
            
            cos_1 = np.dot(vec_ball_car, vec_car_net_self) / (norm_ball_car * norm_car_net_self)
            cos_2 = np.dot(vec_car_ball, vec_net_opp_car) / (norm_car_ball * norm_net_opp_car)
            
            rewards[agent_id] = float(0.5 * cos_1 + 0.5 * cos_2)

        return rewards

class GoalScoredReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in agents}
        
        if state.goal_scored:
            ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
            # Calculate bonus based on ball speed: 1.0 + 0.5 * (v_ball / v_max)
            bonus = 0.5 * (ball_speed / common_values.BALL_MAX_SPEED)
            reward_val = 1.0 + bonus
            
            for agent_id in agents:
                if state.cars[agent_id].team_num == state.scoring_team:
                    rewards[agent_id] = float(reward_val)
                else:
                    rewards[agent_id] = float(-reward_val) # Concede penalty
                    
        return rewards

class DynamicBallTouchReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.lambdas = {}
        self.r_ball = 92.75

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.lambdas = {agent: 1.0 for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            current_lambda = self.lambdas.get(agent_id, 1.0)
            
            if car.ball_touches > 0:
                current_lambda = max(0.1, current_lambda * 0.95)
                self.lambdas[agent_id] = current_lambda
                
                ball_z = state.ball.position[2]
                # Seer formula: lambda * ((ball_z + r_ball) / (2 * r_ball)) ^ 0.2836
                height_factor = ((ball_z + self.r_ball) / (2.0 * self.r_ball)) ** 0.2836
                rewards[agent_id] = float(current_lambda * height_factor)
            else:
                self.lambdas[agent_id] = min(1.0, current_lambda + 0.013)
                rewards[agent_id] = 0.0
                
        return rewards

class DistancePlayerBallReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            dist = float(np.linalg.norm(car.physics.position - state.ball.position))
            rewards[agent_id] = float(np.exp(-0.5 * max(0.0, dist - 92.75) / common_values.CAR_MAX_SPEED))
        return rewards

class ClosestToBallReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {agent_id: 0.0 for agent_id in agents}
        if not agents: return rewards
        dists = {agent_id: float(np.linalg.norm(state.cars[agent_id].physics.position - state.ball.position)) for agent_id in agents}
        closest_agent = min(dists, key=dists.get)
        rewards[closest_agent] = 1.0
        return rewards

class TouchedLastReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_toucher = None
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        for agent_id in agents:
            if state.cars[agent_id].ball_touches > 0:
                self.last_toucher = agent_id
        return {agent_id: 1.0 if agent_id == self.last_toucher else 0.0 for agent_id in agents}

class BehindBallReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            car_y = car.physics.position[1]
            ball_y = state.ball.position[1]
            # If blue (team 0), defend negative Y. So behind ball means car_y < ball_y
            # If orange (team 1), defend positive Y. So behind ball means car_y > ball_y
            if car.team_num == 0:
                rewards[agent_id] = 1.0 if car_y < ball_y else 0.0
            else:
                rewards[agent_id] = 1.0 if car_y > ball_y else 0.0
        return rewards

class VelocityPlayerToBallReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            vel = car.physics.linear_velocity
            speed = float(np.linalg.norm(vel))
            pos_diff = state.ball.position - car.physics.position
            dist = float(np.linalg.norm(pos_diff))
            if speed < 1e-5 or dist < 1e-5:
                rewards[agent_id] = 0.0
                continue
            vel_norm = vel / speed
            dir_norm = pos_diff / dist
            rewards[agent_id] = float(np.dot(vel_norm, dir_norm))
        return rewards

class KickoffReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        ball_pos = state.ball.position
        # Check if ball is at center (Kickoff Y and X are exactly 0, Z is ~93)
        if abs(ball_pos[0]) < 1e-3 and abs(ball_pos[1]) < 1e-3:
            for agent_id in agents:
                car = state.cars[agent_id]
                vel = car.physics.linear_velocity
                speed = float(np.linalg.norm(vel))
                pos_diff = ball_pos - car.physics.position
                dist = float(np.linalg.norm(pos_diff))
                if speed < 1e-5 or dist < 1e-5:
                    rewards[agent_id] = 0.0
                else:
                    rewards[agent_id] = float(np.dot(vel / speed, pos_diff / dist))
        else:
            rewards = {agent_id: 0.0 for agent_id in agents}
        return rewards

class VelocityReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            speed = float(np.linalg.norm(state.cars[agent_id].physics.linear_velocity))
            rewards[agent_id] = float(speed / common_values.CAR_MAX_SPEED)
        return rewards

class BoostAmountReward(RewardFunction):
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            # boost_amount is 0.0 to 1.0 conventionally or 0-100?
            # Adjusting to handle either by bounding it max to 1
            boost = max(0.0, state.cars[agent_id].boost_amount)
            if boost > 1.0: boost /= 100.0 
            rewards[agent_id] = float(np.sqrt(boost))
        return rewards

class ForwardVelocityReward(RewardFunction):
    """Encourages forward movement, and penalizes backward movement. This punishes backwards velocity"""
    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            vel = car.physics.linear_velocity
            forward = car.physics.forward
            rewards[agent_id] = float(np.dot(forward, vel) / common_values.CAR_MAX_SPEED)
        return rewards



class BoostDifferenceReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_boost = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_boost = {agent: float(np.sqrt(min(1.0, initial_state.cars[agent].boost_amount / 100.0 if initial_state.cars[agent].boost_amount > 1.0 else initial_state.cars[agent].boost_amount))) for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            boost = state.cars[agent_id].boost_amount
            if boost > 1.0: boost /= 100.0
            curr_boost_val = float(np.sqrt(boost))
            last_boost_val = self.last_boost.get(agent_id, curr_boost_val)
            rewards[agent_id] = curr_boost_val - last_boost_val
            self.last_boost[agent_id] = curr_boost_val
        return rewards


class DemoReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_demoed = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_demoed = {car_id: initial_state.cars[car_id].is_demoed for car_id in initial_state.cars}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {agent: 0.0 for agent in agents}
        
        for agent_id in agents:
            agent_car = state.cars[agent_id]
            
            # Check all enemies to see if they just crossed into the is_demoed state
            for enemy_id, enemy_car in state.cars.items():
                if enemy_car.team_num != agent_car.team_num:
                    was_demoed = self.last_demoed.get(enemy_id, False)
                    if enemy_car.is_demoed and not was_demoed:
                        # If an enemy became demoed this exact tick, reward our agent
                        rewards[agent_id] += 1.0

        # Update the history for the next tick
        for car_id, car in state.cars.items():
            self.last_demoed[car_id] = car.is_demoed
            
        return rewards


class LandingReward(RewardFunction):
    """Learns to land properly (on wheels) after being in the air. 
    Non-farmable: requires the car to have been airborne for a minimum duration before rewarding."""
    def __init__(self, min_air_ticks: int = 30):
        super().__init__()
        self.min_air_ticks = min_air_ticks
        self.air_ticks = {}
        self.was_on_ground = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.air_ticks = {agent: 0 for agent in agents}
        self.was_on_ground = {agent: True for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            reward = 0.0
            
            is_on_ground = car.on_ground
            was_on_ground = self.was_on_ground.get(agent_id, True)
            
            # Count how long the car is in the air
            if not is_on_ground:
                self.air_ticks[agent_id] = self.air_ticks.get(agent_id, 0) + 1
            else:
                if not was_on_ground:
                    # The car just landed this exact tick
                    if self.air_ticks.get(agent_id, 0) > self.min_air_ticks:
                        # car.physics.up[2] is the Z-component of the car's Up vector.
                        # It is 1.0 if flat on its wheels, -1.0 if upside down on its roof.
                        up_z = float(car.physics.up[2])
                        reward = up_z # High reward for good landing, big penalty for turtle landing
                    
                    # Reset air time counter
                    self.air_ticks[agent_id] = 0
            
            self.was_on_ground[agent_id] = is_on_ground
            rewards[agent_id] = float(reward)
            
        return rewards


class FlipResetReward(RewardFunction):
    """Rewards the agent for successfully achieving a flip reset (getting their flip back in mid-air).
    Non-farmable: Requires the car to be significantly high in the air when the flip state changes."""
    def __init__(self, min_height: float = 300.0):
        super().__init__()
        self.min_height = min_height
        self.had_flip = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.had_flip = {agent: initial_state.cars[agent].has_flip for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            reward = 0.0
            
            # If the car did NOT have a flip last tick, but DOES have a flip this tick
            if car.has_flip and not self.had_flip.get(agent_id, True):
                # And the car is undeniably in the air (not just driving up a wall)
                if not car.on_ground and car.physics.position[2] > self.min_height:
                    reward = 1.0  # Massive payout for a flip reset
                    
            self.had_flip[agent_id] = car.has_flip
            rewards[agent_id] = reward
            
        return rewards


class WaveDashReward(RewardFunction):
    """Rewards the agent for performing a wavedash (flipping immediately before hitting the ground)."""
    def __init__(self):
        super().__init__()
        self.was_flipping = {}

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.was_flipping = {agent: initial_state.cars[agent].is_flipping for agent in agents}

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            reward = 0.0
            
            # If the car just started a flip this exact tick
            if car.is_flipping and not self.was_flipping.get(agent_id, False):
                # If they are extremely close to the ground, but not ON the ground
                if not car.on_ground and car.physics.position[2] < 50.0:
                    reward = 1.0  # Successfully executed a wavedash timing
                    
            self.was_flipping[agent_id] = car.is_flipping
            rewards[agent_id] = reward
            
        return rewards


class AirDribbleCarryReward(RewardFunction):
    """Rewards the agent for keeping the ball close to its nose while both are high in the air."""
    def __init__(self, min_height: float = 400.0, max_distance: float = 200.0):
        super().__init__()
        self.min_height = min_height
        self.max_distance = max_distance

    def reset(self, agents: List[str], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[str], state: GameState, is_terminated: Dict[str, bool], is_truncated: Dict[str, bool], shared_info: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        ball_pos = state.ball.position
        
        for agent_id in agents:
            car = state.cars[agent_id]
            reward = 0.0
            
            # Check if both car and ball are in the air
            if not car.on_ground and car.physics.position[2] > self.min_height and ball_pos[2] > self.min_height:
                dist = float(np.linalg.norm(car.physics.position - ball_pos))
                
                # If the ball is resting very close to the car in mid-air
                if dist < self.max_distance:
                    # Reward scales up the closer the ball is to the car
                    reward = 1.0 - (dist / self.max_distance)
                    
            rewards[agent_id] = reward
            
        return rewards
