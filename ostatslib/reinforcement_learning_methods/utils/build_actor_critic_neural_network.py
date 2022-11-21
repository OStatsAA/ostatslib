"""
build neural network module
"""

from keras import Model as KerasModel
from keras.activations import relu, softmax
from keras.layers import Dense, Input


def build_actor_critic_neural_network(num_hidden_layers: int,
                         num_inputs: int,
                         num_actions: int) -> KerasModel:
    """Build a neural network with num_actions + 1 output for critic estimate

    Args:
        num_hidden_layers (int): number of hidden layers
        num_inputs (int): number of inputs
        num_actions (int): number of actions

    Returns:
        KerasModel: Keras model
    """
    inputs = Input(shape=(num_inputs,))
    common = Dense(num_hidden_layers, activation=relu)(inputs)
    action = Dense(num_actions, activation=softmax)(common)
    critic = Dense(1)(common)

    return KerasModel(inputs=inputs,
                      outputs=[action, critic])
