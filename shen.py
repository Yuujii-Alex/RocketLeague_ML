from typing import Dict, Any

import numpy as np
from rlgym.api import RLGym, StateMutator
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper

from rewards import (
    SpeedTowardBallReward,
    FaceBallReward,
    DistanceBallGoalReward,
    AlignBallGoalReward,
    GoalScoredReward,    
    DynamicBallTouchReward,
    VelocityPlayerToBallReward,
    VelocityReward,
    ForwardVelocityReward,
    TouchedLastReward
)


class DynamicStateMutator(StateMutator[GameState]):
    def __init__(self, kickoff_prob: float = 0.35):
        super().__init__()
        self.kickoff_prob = kickoff_prob
        self.kickoff_mutator = KickoffMutator()

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        if np.random.random() < self.kickoff_prob:
            self.kickoff_mutator.apply(state, shared_info)
            return

        ball_height = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        if ball_height == 0:
            z = np.random.uniform(120, 350)
        elif ball_height == 1:
            z = np.random.uniform(350, 900)
        else:
            z = np.random.uniform(900, 1400)

        state.ball.position = np.array(
            [
                np.random.uniform(-2200, 2200),
                np.random.uniform(-3200, 3200),
                z,
            ],
            dtype=np.float32,
        )
        state.ball.linear_velocity = np.random.uniform(-1200, 1200, 3).astype(
            np.float32
        )
        state.ball.angular_velocity = np.random.uniform(-4, 4, 3).astype(np.float32)

        for car in state.cars.values():
            car.physics.position = np.array(
                [
                    np.random.uniform(-2600, 2600),
                    np.random.uniform(-4200, 4200),
                    17.0,
                ],
                dtype=np.float32,
            )
            car.physics.linear_velocity = np.random.uniform(-700, 700, 3).astype(
                np.float32
            )
            car.physics.angular_velocity = np.random.uniform(-2, 2, 3).astype(
                np.float32
            )
            car.physics.euler_angles = np.array(
                [0.0, np.random.uniform(-np.pi, np.pi), 0.0], dtype=np.float32
            )
            car.boost_amount = np.random.uniform(35, 100)


def build_rlgym_v2_env():
    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0

    action_repeat = 8
    timeout_seconds = 14

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout_seconds=timeout_seconds)

    # Phase 2.1 Reward Structure: Correcting the "Flying/Missing" Rewards
    # Lowering velocity rewards so it doesn't just hold boost and flip blindly.
    # Raising touch reward back up so it actually has to HIT the ball again.
    reward_fn = CombinedReward(
        (GoalScoredReward(), 1.3),
        
        # Raised back up from 0.2 to 0.4. It MUST remember to touch the ball.
        (DynamicBallTouchReward(), 0.4),

        # Give it a steady drip of points for being the person currently in control of the ball.
        (TouchedLastReward(), 0.05),

        (DistanceBallGoalReward(), 0.0025),
        (AlignBallGoalReward(), 0.0025),

        # Severely nerfed the speed rewards. 
        # Making them tie-breakers for "good touches" instead of a raw reason to fly around.
        (VelocityPlayerToBallReward(), 0.001), # Down from 0.01
        (VelocityReward(), 0.0001),            # Down from 0.0005 
        (ForwardVelocityReward(), 0.001),      # Down from 0.01

        (SpeedTowardBallReward(), 0.001),
        (FaceBallReward(), 0.001),

        # Legacy breadcrumbs scaled way down
        (SpeedTowardBallReward(), 0.001),
        (FaceBallReward(), 0.001),
    )

    obs_builder = DefaultObs(
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

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        DynamicStateMutator(),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    import os
    from rlgym_ppo import Learner

    os.environ["WANDB_ENTITY"] = "guoalex.dev"
    os.environ["WANDB_MODE"] = "online"

    n_proc = 11
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        policy_layer_sizes=[1024, 1024, 512, 512],
        critic_layer_sizes=[1024, 1024, 512, 512],
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.015,
        policy_lr=2e-4,
        critic_lr=2e-4,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=100_000,
        timestep_limit=10_000_000_000,
        checkpoints_save_folder="checkpoints",
        checkpoint_load_folder="latest",
        add_unix_timestamp=False,
        log_to_wandb=False,
        wandb_project_name="Shen",
        wandb_run_name="Shen_0.01",
    )

    learner.learn()
