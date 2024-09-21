from attr import dataclass
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from logger.training_impls.discord import DiscordLogger
from parameter_tuning.tune import Tune

import matplotlib

from parameter_tuning.wave_u_net_enhance_transformer import (
    TuneWaveUNetEnhanceTransformer,
)


@dataclass
class Fitter:
    tune: Tune

    def train(self, config):
        matplotlib.use("Agg")  # for ignoring plot

        loss = self.tune.train(config)

        return {"loss": loss}

    def fit(self):
        resources = {"gpu": 1.0}
        # resources = {"cpu": 1.0}
        param_space = self.tune.get_param_space()

        tuner = tune.Tuner(
            tune.with_resources(self.train, resources=resources),
            param_space=param_space,
            run_config=train.RunConfig(
                name="test",
            ),
            tune_config=tune.TuneConfig(
                scheduler=ASHAScheduler(metric="loss", mode="min"),
                num_samples=3,
            ),
        )

        return tuner.fit()


if __name__ == "__main__":
    # fitter = Fitter(TuneExample())
    # fitter = Fitter(TuneWaveUNetEnhanceTwoStageTransformer())
    fitter = Fitter(TuneWaveUNetEnhanceTransformer())

    logger = DiscordLogger()

    logger.on_start({})

    result = fitter.fit()
    print(result.get_best_result(metric="loss", mode="min").config)

    logger.on_finish()
