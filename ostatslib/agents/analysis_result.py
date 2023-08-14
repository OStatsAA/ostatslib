"""
Analysis module
"""

from dataclasses import dataclass, field
from datetime import datetime
from tabulate import tabulate
from ostatslib.actions import ActionInfo
from ostatslib.states import State


StepsRows = list[tuple[int, str, float, str]]


@dataclass(init=True, frozen=True)
class AnalysisResult:
    """
    Analysis class
    """
    initial_state: State
    steps: list[tuple[float, ActionInfo]]
    done: bool
    timestamp: datetime = field(init=False)
    steps_count: int = field(init=False)
    total_reward: float = field(init=False)
    actions_names_list: list[str] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'timestamp', datetime.now())
        object.__setattr__(self, 'steps_count', len(self.steps))
        object.__setattr__(self,
                           'total_reward',
                           sum(reward for reward, _ in self.steps))
        object.__setattr__(self,
                           'actions_names_list',
                           [info.action_name for _, info in self.steps])

    def summary(self) -> str:
        """
        Returns analysis summary

        Returns:
            str: analysis summary
        """
        return (
            f'\nAnalysis executed at {self.timestamp}\n'
            f'Final status is {"Complete" if self.done else "Not Complete"}\n'
            f'Initial State known features:\n{self.__fill_initial_state_row()}\n'
            f'Steps:\n{self.__fill_summary_table_steps_rows()}'
        )

    def __fill_initial_state_row(self):
        diff = self.initial_state - State()
        return tabulate(diff.list_known_features(), tablefmt="plain")

    def __fill_summary_table_steps_rows(self) -> str:
        steps_headers: list[str] = ['Order', 'Step', 'Reward', 'State Change']
        table_rows: StepsRows = []

        for i, (reward, info) in enumerate(self.steps):
            table_rows.append((
                i+1,
                str(info.action_name),
                reward,
                tabulate(self.__get_state_delta(info, i).list_known_features(),
                         tablefmt="plain")
            ))

        return tabulate(table_rows, steps_headers)

    def __get_state_delta(self, info: ActionInfo, i: int):
        if info.next_state:
            if i:
                previous_state = self.steps[i - 1][1].next_state
            else:
                previous_state = self.initial_state

            if previous_state:
                return info.next_state - previous_state

            raise ValueError(
                f'Cannot write State delta, step {i-1} State is None')

        raise ValueError(f'Cannot write State delta, step {i} State is None')
