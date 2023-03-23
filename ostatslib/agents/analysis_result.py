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

        for i, (reward, info) in enumerate(self.steps):
            table_rows.append((
                i+1,
                str(info['action_name']),
                reward,
                tabulate(info['state_delta'].list_known_features(), tablefmt="plain")
            ))

        return tabulate(table_rows, steps_headers)
