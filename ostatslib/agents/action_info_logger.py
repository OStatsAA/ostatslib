"""ActionInfoLogger module
"""

import csv
from stable_baselines3.common.callbacks import BaseCallback

CSV_HEADERS = ['action_name', 'iteration',
               'fit_time', 'raised_exception', 'reward']
CSV_NAME = 'action_info_log.csv'


class ActionInfoLogger(BaseCallback):
    """Logger class to track actions information during training
    """

    _log_file_path: str = CSV_NAME

    def _on_step(self) -> bool:
        zipped_info = self.__get_zipped_info()
        with open(self._log_file_path, 'a', encoding='utf-8') as out:
            csv_out = csv.writer(out)
            csv_out.writerows((list(zipped_info)))
        return super()._on_step()

    def _on_training_start(self) -> None:
        logger_dir = self.locals['callback'].logger.dir
        if logger_dir:
            self._log_file_path = logger_dir + CSV_NAME

        with open(self._log_file_path, 'w', encoding='utf-8') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(CSV_HEADERS)

    def __get_zipped_info(self):
        actions_names: list[str] = []
        fit_times: list[float | None] = []
        raised_exceptions: list[bool] = []
        for info in self.locals['infos']:
            actions_names.append(info.action_name)
            fit_times.append(info.fit_time)
            raised_exceptions.append(info.raised_exception)

        rewards: list[float] = self.locals['rewards']
        iterations: list[int] = [self.locals['iteration']] * len(rewards)
        return zip(actions_names, iterations, fit_times, raised_exceptions, rewards)
