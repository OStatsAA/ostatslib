"""
ActorCritic module
"""

import numpy as np
import tensorflow as tf
from pandas import DataFrame
from keras.losses import Huber
from keras.optimizers import Adam

from ostatslib.environments import Environment
from ostatslib.states import State

from .reinforcement_learning_method import ReinforcementLearningMethod
from .utils import (build_actor_critic_neural_network, convert_state_to_tensor,
                    ModelNotFitError, normalize_values_list)


class ActorCritic(ReinforcementLearningMethod):
    """
    Actor-Critic implementation
    """

    def __init__(self, num_hidden_layers: int = 128, discount_rate: float = 0.99) -> None:
        self.__num_hidden_layers = num_hidden_layers
        self.__discount_rate = discount_rate
        self.__neural_network = None
        self.__is_fit: bool = False

    @property
    def is_fit(self) -> bool:
        return self.__is_fit

    def run_analysis(self,
                     state: State,
                     data: DataFrame,
                     environment: Environment,
                     max_steps: int) -> tuple[list, bool]:
        if not self.__is_fit:
            raise ModelNotFitError()

        steps = []
        for _ in range(1, max_steps):
            action_result, done = self.__run_analysis_step(data,
                                                           environment,
                                                           state)
            state = action_result.state
            steps.append(action_result)

            if done:
                break

        return steps, done

    def run_training_episode(self,
                             state: State,
                             data: DataFrame,
                             environment: Environment,
                             max_steps: int) -> float:
        action_probs_history, critic_value_history, rewards_history = [], [], []
        episode_reward = 0
        num_actions = len(environment.actions_space.actions_names_list)
        self.__ensure_neural_network_is_initialized(len(state.features_vector),
                                                    num_actions)

        with tf.GradientTape() as tape:
            for _ in range(1, max_steps):
                action_result, done = self.__run_training_step(data,
                                                               environment,
                                                               action_probs_history,
                                                               critic_value_history,
                                                               state)

                state = action_result.state
                rewards_history.append(action_result.reward)
                episode_reward += action_result.reward

                if done:
                    break

            actor_losses, critic_losses = self.__calc_losses(action_probs_history,
                                                             critic_value_history,
                                                             rewards_history)

            self.__back_propagate_losses(tape,
                                         actor_losses,
                                         critic_losses)
        self.__is_fit = True
        return float(episode_reward)

    def __run_training_step(self,
                            data: DataFrame,
                            environment: Environment,
                            action_probs_history: list,
                            critic_value_history: list,
                            state: State):
        action_probs, critic_value = self.get_actions_probs_and_critic(state)
        critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        action = np.random.choice(action_probs.shape[1],
                                  p=np.squeeze(action_probs))
        action_probs_history.append(tf.math.log(action_probs[0, action]))

        # Apply the sampled action in our environment
        action_result, done = environment.run_action(state,
                                                     data,
                                                     action)

        return action_result, done

    def get_actions_probs_and_critic(self, state: State) -> tuple[tf.Tensor, float]:
        """
        Get actions probabilities and critic value for a given state

        Args:
            state (State): state

        Returns:
            tuple[tf.Tensor, float]: actions probabilities and critic reward estimate
        """
        state_tensor = convert_state_to_tensor(state)
        action_probs, critic_value = self.__neural_network(state_tensor)
        return action_probs, critic_value
    
    def fit(self) -> None:
        pass

    def __run_analysis_step(self,
                            data: DataFrame,
                            environment: Environment,
                            state: State):
        action_probs, _ = self.get_actions_probs_and_critic(state)
        action = np.argmax(action_probs)

        action_result, done = environment.run_action(state, data, action)

        return action_result, done

    def __ensure_neural_network_is_initialized(self, num_states_features, num_actions):
        if not self.__neural_network:
            self.__neural_network = build_actor_critic_neural_network(self.__num_hidden_layers,
                                                         num_states_features,
                                                         num_actions)

    def __calc_losses(self,
                      action_probs_history: list,
                      critic_value_history: list,
                      rewards_history: list):
        returns = self.__calculate_discounted_returns_normalized(
            rewards_history)
        huber_loss = Huber()
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []

        for log_prob, critic_value, reward in history:
            diff = reward - critic_value
            actor_losses.append(-log_prob * diff)  # actor loss
            critic_losses.append(huber_loss(tf.expand_dims(critic_value, 0),
                                            tf.expand_dims(reward, 0)))

        return actor_losses, critic_losses

    def __calculate_discounted_returns_normalized(self, rewards_history: list):
        returns = []
        discounted_sum = 0
        for reward in rewards_history[::-1]:
            discounted_sum = reward + self.__discount_rate * discounted_sum
            returns.insert(0, discounted_sum)
        return normalize_values_list(returns)

    def __back_propagate_losses(self,
                                tape: tf.GradientTape,
                                actor_losses,
                                critic_losses):
        optimizer = Adam(learning_rate=0.01)
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(
            loss_value, self.__neural_network.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, self.__neural_network.trainable_variables))
