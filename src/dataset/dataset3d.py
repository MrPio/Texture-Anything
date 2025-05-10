import abc
import pandas as pd


class Dataset3D(abc.ABC):
    @property
    @abc.abstractmethod
    def annotations(self) -> pd.DataFrame | None:
        """The metadata provided by the authors of the dataset, if available"""
        ...

    @property
    @abc.abstractmethod
    def statistics(self) -> pd.DataFrame: 
        """The statistics generated on the downloaded models. Columns are: `meshCount`, `uvCount`, `diffuseCount`"""
        ...

    @property
    @abc.abstractmethod
    def paths(self) -> dict[str, str]:
        """A `dict` with UID as key and object path as value"""
        ...

    def download(self) -> None:
        raise NotImplementedError()