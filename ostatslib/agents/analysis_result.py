"""
Analysis module
"""

from dataclasses import dataclass, field
from datetime import datetime
from tabulate import tabulate
from ostatslib.actions import ActionResult
from ostatslib.states import State


StepsRows = list[tuple[int, str, float, str]]


@dataclass(init=True, frozen=True)
class AnalysisResult:
    """
    Analysis class
    """
    initial_state: State
    steps: list[ActionResult]
    done: bool
    timestamp: datetime = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'timestamp', datetime.now())

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

        for i, (state, reward, info) in enumerate(self.steps):
            table_rows.append((
                i+1,
                str(info['action_name']),
                reward,
                self.__get_steps_diff_table(i, state)
            ))

        return tabulate(table_rows, steps_headers)

    def __get_steps_diff_table(self, i: int, state: State) -> str:
        previous_state = self.steps[i-1][0] if i else self.initial_state
        diff = state - previous_state

        diff_table = tabulate(diff.list_known_features(), tablefmt="plain")
        return diff_table
