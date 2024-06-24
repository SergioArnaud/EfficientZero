import os
import csv
import uuid
import time
import numpy as np
from core.game import Game
from core.utils import arr_to_str


EXPERIMENT_ID = "{}".format(time.strftime("%Y.%m.%d_%H.%M.%S"))

class AtariWrapper(Game):
    def __init__(
        self,
        env,
        discount: float,
        cvt_string=True,
        train_env=False,
        test_env=False,
        test_name="generic",
        game_name=''
    ):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)

        date = time.strftime("%Y.%m.%d")
        self.experiment_uuid = uuid.uuid1()
        self.game_name = game_name
        # experiment_id = "{}_{}".format(time.strftime("%Y.%m.%d_%H.%M.%S"), game_name)
        self.train_env = train_env
        self.steps = 0

        self.experiment_outpath = "../experiments/{}/{}/{}/{}".format(
            "EfficientZero", game_name, date, EXPERIMENT_ID,
        )
        self.experiment_tests_outpath = "{}/test-{}".format(
            self.experiment_outpath,
            test_name,
        )

        self.history = []

        # NOTE(JP): This actually works.
        if train_env:
            os.makedirs(self.experiment_outpath, exist_ok=True)
            with open(
                "{}/{}_{}_reward_history.csv".format(
                    self.experiment_outpath,
                    self.experiment_uuid,
                    self.game_name
                ),
                "w",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["steps", "reward", "done", "info"]
                )

        self.cvt_string = cvt_string

        # NOTE(JP): [DEV] From here down is experimental.
        if train_env:
            # TODO(JP): should I add train prefix to the filename.
            init_reward_history(
                path=self.experiment_outpath,
                filename = "{}_{}_train_reward_history.csv".format(
                    self.experiment_uuid, self.game_name
                ),
            )

        if test_env:
            init_reward_history(
                path=self.experiment_tests_outpath,
                filename = "{}_{}_test_reward_history.csv".format(
                    self.experiment_uuid, self.game_name
                ),
            )

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):

        self.steps += 1
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)

        # Count real number of frames processed.
        real_frame_count = self.steps * 4
        info["real_frame_count"] = real_frame_count

        if self.cvt_string:
            observation = arr_to_str(observation)

        if self.train_env:
            with open(
                "{}/{}_{}_reward_history.csv".format(
                    self.experiment_outpath,
                    self.experiment_uuid,
                    self.game_name
                ),
                "a",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([self.steps, reward, done, info])

        # NOTE(JP): [DEV] From here down is experimental.
        if self.train_env:
            write_step_to_csv(
                self.experiment_outpath,
                "{}_{}_train_reward_history.csv".format(
                    self.experiment_uuid, self.game_name
                ),
                self.steps,
                reward,
                done,
                info,
            )

        if self.test_env:
            self.history = [self.steps, reward, done, info]
            write_step_to_csv(
                self.experiment_outpath,
                "{}_{}_test_reward_history.csv".format(
                    self.experiment_uuid, self.game_name
                ),
                self.steps,
                reward,
                done,
                info,
            )

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()


# filename = "{}_{}_reward_history.csv".format(experiment_uuid, game_name)
def init_reward_history(path, filename):
    os.makedirs(path, exist_ok=True)
    with open("{}/{}".format(path, filename), "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["steps", "reward", "done", "info"]
        )

def write_step_to_csv(path, filename, step, reward, done, info):
    with open("{}/{}".format(path, filename), "a") as file:
        writer = csv.writer(file)
        writer.writerow([step, reward, done, info])