import ostatslib.actions.regressors.linear_models as linear_models
from statsforecast.models import AutoARIMA
from datasetsforecast.m4 import M4
from ostatslib.config import DEFAULT_CONFIG

from ostatslib.states import State

def test_auto_arima_regression() -> None:
    auto_arima = linear_models.AutoARIMARegression()
    m4_datasets = M4().load(directory='./.m4_cache/', group='Yearly')
    data = m4_datasets[0].query("unique_id == 'Y1'")
    state = State()
    state.set('time_convertible_variable', 'ds')
    state.set('response_variable_label', 'y')
    state, reward, info = auto_arima.execute(data, state, config=DEFAULT_CONFIG)
    assert reward > 0
    assert isinstance(info.model, AutoARIMA)
    assert float(state.get(auto_arima.action_key + '_score_reward')) > 0
