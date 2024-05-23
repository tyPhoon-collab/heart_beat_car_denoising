import os
from dataset.dataset import NoisyHeartbeatDataset
from dataset.loader import MatLoader
from dataset.randomizer import Randomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from utils.gain_controller import GainController


class DatasetFactory:
    @classmethod
    def create(
        cls,
        clean_file_path: str,
        noisy_file_path: str,
        sample_rate: int | None = None,
        sample_rate_map: tuple[int, int] | None = None,
        randomizer: Randomizer | None = None,
        gain_controller: GainController | None = None,
        train=True,
    ):
        assert sample_rate is not None or sample_rate_map is not None

        clean_data_loader = cls.get_loader(clean_file_path)
        noisy_data_loader = cls.get_loader(noisy_file_path)

        return NoisyHeartbeatDataset(
            clean_data=clean_data_loader.load()["ch1z"].to_numpy(),
            noisy_data=noisy_data_loader.load()["ch1z"].to_numpy(),
            sampling_rate_converter=(
                ScipySamplingRateConverter(
                    input_rate=sample_rate,  # type: ignore
                    output_rate=sample_rate,  # type: ignore
                )
                if sample_rate_map is None
                else ScipySamplingRateConverter(
                    input_rate=sample_rate_map[0],
                    output_rate=sample_rate_map[1],
                )
            ),
            randomizer=randomizer,
            gain_controller=gain_controller,
            train=train,
        )

    @classmethod
    def create_240219(cls, **kwargs):
        return cls.create(
            clean_file_path="data/240219_Rawdata/Stop.mat",
            noisy_file_path="data/240219_Rawdata/100km.mat",
            sample_rate_map=(32000, 1000),
            **kwargs,
        )

    @classmethod
    def create_240517(cls, **kwargs):
        return cls.create(
            clean_file_path="data/240517_Rawdata/HS_data_serial.mat",
            noisy_file_path="data/240517_Rawdata/Noise_data_serial.mat",
            sample_rate=1000,
            **kwargs,
        )

    @classmethod
    def get_loader(cls, path: str):
        """
        先方からのデータ形式が毎回異なる。差分を吸収してMatLoaderとして扱えるようにする関数
        """
        if "240517" in path:
            print("240517 loader is chosen")
            # "data/240517_Rawdata/HS_data.mat" -> "HS_data"
            filename = os.path.splitext(os.path.basename(path))[0]

            # 最後の '_' の位置を見つける
            index = filename.rfind("_")

            # 最初からその位置までをスライス
            data_key = filename[:index]

            return MatLoader(
                file_path=path,
                columns=["ch1z"],
                data_key=data_key,
            )
        elif "240219" in path:
            print("240517 loader is chosen")

            return MatLoader(
                file_path=path,
                columns=["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
                data_key="data",
            )
        else:
            raise ValueError(f"Unsupported data path: {path}")
