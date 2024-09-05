import os
from typing import Callable

import numpy as np
from dataset.dataset import NoisyHeartbeatDataset
from dataset.filter import FIRBandpassFilter
from dataset.loader import ConcatColumnsLoader, MatLoader, cacheable
from dataset.sampling_rate_converter import (
    NoSamplingRateConverter,
    ScipySamplingRateConverter,
)


class DatasetFactory:
    @classmethod
    def create(
        cls,
        clean_file_path: str,
        noisy_file_path: str,
        clean_data_modifier: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        noisy_data_modifier: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        sample_rate: int | None = None,
        sample_rate_map: tuple[int, int] | None = None,
        train=True,
        **kwargs,
    ):
        assert sample_rate is not None or sample_rate_map is not None

        clean_data_loader = cls.build_loader(clean_file_path)
        noisy_data_loader = cls.build_loader(noisy_file_path)

        clean_data = cls.load(clean_data_modifier, clean_data_loader)
        noisy_data = cls.load(noisy_data_modifier, noisy_data_loader)

        return NoisyHeartbeatDataset(
            clean_data=clean_data,
            noisy_data=noisy_data,
            sampling_rate_converter=(
                NoSamplingRateConverter(rate=sample_rate)  # type: ignore
                if sample_rate_map is None
                else ScipySamplingRateConverter(
                    input_rate=sample_rate_map[0],
                    output_rate=sample_rate_map[1],
                )
            ),
            train=train,
            **kwargs,
        )

    @classmethod
    def create_240219(cls, base_dir: str = "", **kwargs):
        c = os.path.join(base_dir, "data", "240219_Rawdata", "Stop.mat")
        n = os.path.join(base_dir, "data", "240219_Rawdata", "100km.mat")
        return cls.create(
            clean_file_path=c,
            noisy_file_path=n,
            sample_rate_map=(32000, 1000),
            **kwargs,
        )

    @classmethod
    def create_240517(cls, base_dir: str = "", **kwargs):
        c = os.path.join(base_dir, "data", "240517_Rawdata", "HS_data_serial.mat")
        n = os.path.join(base_dir, "data", "240517_Rawdata", "Noise_data_serial.mat")
        return cls.create(
            clean_file_path=c,
            noisy_file_path=n,
            sample_rate=1000,
            **kwargs,
        )

    @classmethod
    def create_240517_filtered(cls, base_dir: str = "", **kwargs):
        c = os.path.join(base_dir, "data", "240517_Rawdata", "HS_data_serial.mat")
        n = os.path.join(base_dir, "data", "240517_Rawdata", "Noise_data_serial.mat")
        modifier = FIRBandpassFilter((25, 55), 1000).apply

        return cls.create(
            clean_file_path=c,
            noisy_file_path=n,
            clean_data_modifier=modifier,
            noisy_data_modifier=modifier,
            sample_rate=1000,
            **kwargs,
        )

    @classmethod
    def create_240826_filtered(cls, base_dir: str = "", **kwargs):
        c = os.path.join(base_dir, "data", "240826_Rawdata", "HS_data_stop.mat")
        n = os.path.join(base_dir, "data", "240826_Rawdata", "Noise_data_100km.mat")
        modifier = FIRBandpassFilter((25, 55), 1000).apply

        return cls.create(
            clean_file_path=c,
            noisy_file_path=n,
            clean_data_modifier=modifier,
            noisy_data_modifier=modifier,
            sample_rate=1000,
            **kwargs,
        )

    @classmethod
    def load(cls, modifier, loader):
        return modifier(loader.load()["ch1z"].to_numpy())

    @classmethod
    @cacheable
    def build_loader(cls, path: str):
        """
        先方からのデータ形式が毎回異なる。差分を吸収してMatLoaderとして扱えるようにする関数
        """
        if "240219" in path:
            print("240219 loader is chosen")

            return MatLoader(
                file_path=path,
                columns=["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
                data_key="data",
            )
        elif "240517" in path:
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

        elif "240826" in path:
            print("240826 loader is chosen")

            if "ECG" in path:
                return ConcatColumnsLoader(
                    loader=MatLoader(
                        file_path=path,
                        data_key="ECG_data",
                    )
                )
            elif "HS" in path:
                return ConcatColumnsLoader(
                    loader=MatLoader(
                        file_path=path,
                        data_key="HS_data",
                    )
                )
            elif "Noise" in path:
                return ConcatColumnsLoader(
                    loader=MatLoader(
                        file_path=path,
                        data_key="data",
                    )
                )
            else:
                raise ValueError(f"Unsupported data path: {path}")
        else:
            raise ValueError(f"Unsupported data path: {path}")
