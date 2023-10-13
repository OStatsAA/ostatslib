"""ActionsSpace module
"""

import inspect
from types import ModuleType
from functools import cached_property

from gymnasium.spaces import MultiBinary
import numpy as np
import numpy.typing as npt

from ostatslib.actions.base import Action

from . import exploratory_actions, classifiers, regressors, clustering

MaskNDArray = npt.NDArray[np.int8]
ENCODING_LENGTH = 7

_CLASSIFIERS_OFFSET = 32
_REGRESSION_OFFSET = 64
_CLUSTERING_OFFSET = 96
_LAST_KEY = 127
_NONES_DICT = {key: None for key in range(_LAST_KEY + 1)}


def _binary_to_integer(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))


def _is_action(type_: type):
    return inspect.isclass(type_) and issubclass(type_, Action)


def _build_actions_dict(actions_module: ModuleType, offset: int = 0) -> dict[int, Action]:
    actions = inspect.getmembers(actions_module, _is_action)
    return {index + offset: action() for index, (_, action) in enumerate(actions)}


class ActionsSpace(MultiBinary):
    """Actions space initializes actions and make them available to 
    the environment. Extends Gymnasium MultiBinary space.
    """

    def __init__(self) -> None:
        _exploratory_actions = _build_actions_dict(exploratory_actions)
        _classifiers = _build_actions_dict(classifiers, _CLASSIFIERS_OFFSET)
        _regressors = _build_actions_dict(regressors, _REGRESSION_OFFSET)
        _clustering = _build_actions_dict(clustering, _CLUSTERING_OFFSET)
        self._actions = _NONES_DICT | _exploratory_actions | _classifiers | _regressors | _clustering
        super().__init__(ENCODING_LENGTH)

    @property
    def actions_dict(self) -> dict[int, Action | None]:
        """Actions in actions space

        Returns:
            dict[int, Action | None]: dictionary of actions in actions space
        """
        return self._actions

    @cached_property
    def actions_list(self) -> list[Action]:
        """List of valid actions available in actions space

        Returns:
            list[Action]: list of available actions
        """
        return [action for action in self.actions_dict.values() if action is not None]

    @property
    def encoding_length(self) -> int:
        """
        Returns encoding length (# of digits in the encoding)

        Returns:
            int: # of digits in the encoding
        """
        return ENCODING_LENGTH

    def get_action(self, numeric_key: int | np.ndarray) -> Action | None:
        """Get action by numeric key.
        Numeric key may be an integer or ndarray binary representation (from policy network)

        Args:
            numeric_key (int | np.ndarray): action numeric key

        Returns:
            Action | None: action or None if dict key is not an action
        """
        if isinstance(numeric_key, np.ndarray):
            numeric_key = _binary_to_integer(numeric_key)

        return self._actions[numeric_key]

    def get_action_by_class(self, action: type) -> Action:
        """Get action instance in actions space by class

        Args:
            action (type): action type

        Raises:
            ValueError: raised if no action of type is found

        Returns:
            Action: action instance in actions space
        """
        for action_ in self.actions_list:
            if isinstance(action_, action):
                return action_

        raise ValueError(f'No class matched class {action}.')

    def sample(self, mask: MaskNDArray | None = None) -> np.ndarray:
        index = np.random.choice(len(self))
        return np.unpackbits(np.array(index, dtype='uint8'), count=self.encoding_length)

    def __len__(self):
        return len(self._actions)
