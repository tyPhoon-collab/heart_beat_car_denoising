from config import TrainConfig
from utils.model_save_validator import (
    AnyCompositeModelSaveValidator,
    BestModelSaveValidator,
    ModelSaveValidator,
    SpecificEpochModelSaveValidator,
)
from utils.model_saver import ModelSaver, WithDateModelSaver, WithIdModelSaver


class ModelSaverFactory:
    @classmethod
    def config(cls, c: TrainConfig) -> tuple[ModelSaver, ModelSaveValidator]:
        id = c.id
        path = c.checkpoint_path
        model_saver = (
            WithDateModelSaver(base_directory=path)
            if id is None
            else WithIdModelSaver(base_directory=path, id=id)
        )

        best_considered_epoch_from = (
            c.progressive_end_epoch + 1 if c.progressive_gain else 1
        )

        model_save_validator = AnyCompositeModelSaveValidator(
            validators=[
                BestModelSaveValidator(epoch_index_from=(best_considered_epoch_from)),
                SpecificEpochModelSaveValidator.last(c.epoch),
            ]
        )
        return model_saver, model_save_validator
