import inspect
from types import ModuleType
from functools import cache, cached_property

from gymnasium.spaces import MultiBinary
import numpy as np
import numpy.typing as npt

from ostatslib.actions.base import Action

from . import exploratory_actions, classifiers, regressors

MaskNDArray = npt.NDArray[np.int8]
ENCODING_LENGTH = 6

_CLASSIFIERS_OFFSET = 16
_REGRESSION_OFFSET = 32
_LAST_KEY = 63
_NONES_DICT = {key: None for key in range(_LAST_KEY + 1)}


def _binary_to_integer(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))


def _is_action(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action)


def _build_actions_dict(actions_module: ModuleType, offset: int = 0) -> dict[int, Action]:
    actions = inspect.getmembers(actions_module, _is_action)
    return {index + offset: action() for index, (_, action) in enumerate(actions)}


class ActionsSpace(MultiBinary):

    def __init__(self) -> None:
        _exploratory_actions = _build_actions_dict(exploratory_actions)
        _classifiers = _build_actions_dict(classifiers, _CLASSIFIERS_OFFSET)
        _regressors = _build_actions_dict(regressors, _REGRESSION_OFFSET)
        self._actions = _NONES_DICT | _exploratory_actions | _classifiers | _regressors
        super().__init__(ENCODING_LENGTH)

    @property
    def actions_dict(self) -> dict[int, Action | None]:
        return self._actions

    @cached_property
    def actions_list(self) -> list[Action]:
        return [action for action in self.actions_dict.values() if action is not None]

    @property
    def encoding_length(self) -> int:
        """
        Returns encoding length (# of digits in the)

        Returns:
            int: # of digits in the encoding
        """
        return ENCODING_LENGTH

    def get_action(self, numeric_key: int | np.ndarray) -> Action | None:
        if isinstance(numeric_key, np.ndarray):
            numeric_key = _binary_to_integer(numeric_key)

        return self._actions[numeric_key]

    @cache
    def get_action_by_class(self, action_class: Action) -> Action:
        for action in self.actions_list:
            if action is action_class:
                return action

        raise ValueError(f'No class matched class {action_class}.')

    def sample(self, mask: MaskNDArray | None = None) -> np.ndarray:
        index = np.random.choice(len(self))
        return np.unpackbits(np.array(index, dtype='uint8'), count=self.encoding_length)

    def __len__(self):
        return len(self._actions)
