import os
import torch
import numpy as np
from rlgym.rocket_league.api import GameState, Car, PhysicsObject, GameConfig
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league import common_values
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class ShenBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.obs_builder = None
        self.action_parser = None
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_repeat = 8
        self.tick_skip_counter = 0
        self.prev_action = np.zeros(8)
        self.lookup_table = None

    def initialize_agent(self):
        # Initialize Observation Builder (must match shen.py)
        self.obs_builder = DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray(
                [
                    1 / common_values.SIDE_WALL_X,
                    1 / common_values.BACK_NET_Y,
                    1 / common_values.CEILING_Z,
                ]
            ),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            boost_coef=1 / 100.0,
        )

        # Initialize Action Parser
        self.lookup_table = LookupTableAction()

        # Load Policy
        model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_policy.pt")

        # Fallback to latest checkpoint if models/ppo_policy.pt is missing
        if not os.path.exists(model_path):
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [
                    d
                    for d in os.listdir(checkpoint_dir)
                    if os.path.isdir(os.path.join(checkpoint_dir, d))
                ]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=int)[-1]
                    model_path = os.path.join(
                        checkpoint_dir, latest_checkpoint, "PPO_POLICY.pt"
                    )
                    self.logger.info(f"Using latest checkpoint: {model_path}")

        if os.path.exists(model_path):
            try:
                loaded_obj = torch.load(model_path, map_location=self.device)

                # Check if it's a state dict or a full module
                if isinstance(loaded_obj, torch.nn.Module):
                    self.policy = loaded_obj
                elif isinstance(loaded_obj, dict):
                    self.logger.info(
                        f"Detected state_dict in {model_path}. Rebuilding policy..."
                    )
                    # Rebuild the policy architecture (must match shen.py)
                    from rlgym_ppo.ppo import DiscreteFF

                    # Determine dimensions (can be inferred from state_dict)
                    input_shape = loaded_obj["model.0.weight"].shape[1]
                    n_actions = loaded_obj[list(loaded_obj.keys())[-2]].shape[
                        0
                    ]  # Usually the last weight/bias

                    self.policy = DiscreteFF(
                        input_shape, n_actions, [1024, 1024, 512, 512], self.device
                    )
                    self.policy.load_state_dict(loaded_obj)
                else:
                    self.logger.error(
                        f"Loaded object from {model_path} is an unknown type: {type(loaded_obj)}"
                    )
                    self.policy = None

                if self.policy is not None:
                    self.policy.to(self.device)
                    self.policy.eval()
                    self.logger.info(
                        f"Successfully initialized policy from {model_path}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to load policy: {e}")
                import traceback

                self.logger.error(traceback.format_exc())
                self.policy = None
        else:
            self.logger.error(f"Could not find any policy file. Bot will not move.")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.policy is None:
            return SimpleControllerState()

        # Handle action repeat logic
        if self.tick_skip_counter % self.action_repeat != 0:
            self.tick_skip_counter += 1
            return self.get_controller_state(self.prev_action)

        self.tick_skip_counter = 1

        # 1. Convert Packet to RLGym 2.0 GameState
        state = self.parse_packet(packet)

        # 2. Build Observation
        # DefaultObs.build_obs expects (agents, state, shared_info)
        obs = self.obs_builder.build_obs([self.index], state, {})[self.index]

        # 3. Inference
        with torch.no_grad():
            input_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            # Depending on how rlgym-ppo saves the model, this might be a full model or just the actor
            # Most likely it's a DiscretePolicy or similar
            action_idx = self.policy.get_action(input_tensor)[0].cpu().numpy()[0]

        # 4. Map Action Index to Controls
        # LookupTableAction.parse_actions returns a dict of actions (possibly with a tick dimension)
        parsed_actions = self.lookup_table.parse_actions(
            {self.index: np.array([action_idx])}, state, {}
        )
        actions = parsed_actions[self.index]
        if len(actions.shape) > 1:
            actions = actions[0]

        self.prev_action = actions

        return self.get_controller_state(actions)

    def parse_packet(self, packet: GameTickPacket) -> GameState:
        # Construct Physics Objects
        ball = self.parse_physics(packet.game_ball.physics)

        cars = {}
        for i in range(packet.num_cars):
            car_data = packet.game_cars[i]
            car = Car()
            car.team_num = car_data.team
            car.hitbox_type = 0  # Default
            car.ball_touches = 0
            car.bump_victim_id = None
            car.demo_respawn_timer = 3.0 if car_data.is_demolished else 0.0
            car.wheels_with_contact = (
                car_data.has_wheel_contact,
                car_data.has_wheel_contact,
                car_data.has_wheel_contact,
                car_data.has_wheel_contact,
            )
            car.supersonic_time = 1.0 if car_data.is_super_sonic else 0.0
            car.boost_amount = car_data.boost
            car.boost_active_time = 0.0
            car.handbrake = 0.0
            car.is_jumping = car_data.jumped
            car.has_jumped = car_data.jumped
            car.is_holding_jump = False
            car.jump_time = 0.0
            car.has_flipped = car_data.double_jumped
            car.has_double_jumped = car_data.double_jumped
            car.air_time_since_jump = 0.0
            car.flip_time = 0.0
            car.flip_torque = np.zeros(3)
            car.is_autoflipping = False
            car.autoflip_timer = 0.0
            car.autoflip_direction = 0.0
            car.physics = self.parse_physics(car_data.physics)
            car._inverted_physics = None
            cars[i] = car

        # Boost Pads
        boost_pad_timers = np.zeros(packet.num_boost)
        for i in range(packet.num_boost):
            boost_pad_timers[i] = 0.0 if packet.game_boosts[i].is_active else 10.0

        game_config = GameConfig()
        game_config.gravity = packet.game_info.world_gravity_z
        game_config.boost_consumption = 1.0
        game_config.dodge_deadzone = 0.5

        state = GameState()
        state.tick_count = packet.game_info.frame_num
        state.goal_scored = False
        state.config = game_config
        state.cars = cars
        state.ball = ball
        state._inverted_ball = None
        state.boost_pad_timers = boost_pad_timers
        state._inverted_boost_pad_timers = None

        return state

    def parse_physics(self, physics) -> PhysicsObject:
        pos = np.array([physics.location.x, physics.location.y, physics.location.z])
        vel = np.array([physics.velocity.x, physics.velocity.y, physics.velocity.z])
        ang_vel = np.array(
            [
                physics.angular_velocity.x,
                physics.angular_velocity.y,
                physics.angular_velocity.z,
            ]
        )
        euler = np.array(
            [physics.rotation.pitch, physics.rotation.yaw, physics.rotation.roll]
        )

        obj = PhysicsObject()
        obj.position = pos
        obj.linear_velocity = vel
        obj.angular_velocity = ang_vel
        obj._quaternion = None
        obj._rotation_mtx = None
        obj._euler_angles = euler
        return obj

    def get_controller_state(self, actions) -> SimpleControllerState:
        # RLGym 2.0 LookupTableAction returns [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        controls = SimpleControllerState()
        controls.throttle = float(actions[0])
        controls.steer = float(actions[1])
        controls.pitch = float(actions[2])
        controls.yaw = float(actions[3])
        controls.roll = float(actions[4])
        controls.jump = bool(actions[5] > 0.5)
        controls.boost = bool(actions[6] > 0.5)
        controls.handbrake = bool(actions[7] > 0.5)
        return controls
