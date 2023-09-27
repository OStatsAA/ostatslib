from pandas import DataFrame
import pmlb


from ostatslib.actions.classifiers import ElasticNetLogisticRegressionClassification
from ostatslib.config import DEFAULT_CONFIG
from ostatslib.environments.data_generators.pmlb_generator import PMLB_CACHE_FOLDER
from ostatslib.states import State


def test_target_model_estimator_timeout() -> None:
    action = ElasticNetLogisticRegressionClassification()
    data: DataFrame = pmlb.fetch_data('mnist', local_cache_dir=PMLB_CACHE_FOLDER)
    config = DEFAULT_CONFIG
    config['FIT_TIMEOUT'] = 10
    state = State()
    state.set('response_variable_label', 'target')
    state.set('is_response_discrete', 1)
    state.set('response_unique_values_ratio', 0.001)
    state.set('log_rows_count', 0.5)

    _, reward, info = action.execute(data, state, config)
    assert reward < 0
    assert info.raised_exception
