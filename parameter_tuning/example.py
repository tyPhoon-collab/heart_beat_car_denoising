from typing import Any
from parameter_tuning.tune import Loss, Tune
from ray import tune


class TuneExample(Tune):
    def get_param_space(self) -> dict[str, Any]:
        return {
            "a": tune.grid_search([0, 1, 2, 3]),
            "b": tune.grid_search([0, 5, 10]),
        }

    def train(self, config) -> Loss:
        return config["a"] * 5 + config["b"]
