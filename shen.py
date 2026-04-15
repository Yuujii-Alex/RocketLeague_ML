def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition,
        NoTouchTimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
        GoalReward,
        TouchReward,
    )
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence,
        FixedTeamSizeMutator,
        KickoffMutator,
    )
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    import numpy as np

    from rewards import InAirReward, SpeedTowardBallReward, FaceBallReward

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout_seconds=timeout_seconds)

    reward_fn = CombinedReward(
        (TouchReward(), 50),
        (GoalReward(), 10.0),
        (InAirReward(), 0.15),
        (SpeedTowardBallReward(), 5),
        (FaceBallReward(), 1),
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
        KickoffMutator(),
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
    from rlgym_ppo import Learner
    import os

    os.environ["WANDB_ENTITY"] = "guoalex.dev"
    # 6 CPU processes
    n_proc = 12

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,  # Leave this empty for now.
        ppo_batch_size=50_000,  # batch size - much higher than 300K doesn't seem to help most people
        policy_layer_sizes=[1024, 1024, 512, 512],  # leaner policy network
        critic_layer_sizes=[1024, 1024, 512, 512],  # leaner critic network
        ts_per_iteration=50_000,  # timesteps per training iteration - set this equal to the batch size
        exp_buffer_size=150_000,  # size of experience buffer - keep this 2 - 3x the batch size
        ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
        ppo_ent_coef=0.01,  # entropy coefficient - resetting to 0.01 for fresh exploration
        policy_lr=2e-4,  # policy learning rate
        critic_lr=2e-4,  # critic learning rate
        ppo_epochs=2,  # number of PPO epochs
        standardize_returns=True,  # Don't touch these.
        standardize_obs=False,  # Don't touch these.
        save_every_ts=100_000,  # save every 1M steps
        timestep_limit=1_000_000_000,  # Train for 1B steps
        checkpoints_save_folder="checkpoints",  # Save checkpoints in a new folder
        checkpoint_load_folder="latest",  # latest or None
        add_unix_timestamp=False,
        log_to_wandb=True,  # Set this to True if you want to use Weights & Biases for logging.
        wandb_project_name="ShenV2",  # New project for a clean graph
    )
    learner.learn()
