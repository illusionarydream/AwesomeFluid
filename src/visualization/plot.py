from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


class EasyLinePlot:
    """
    A simple and pythonic line-plot helper.

    Data model:
        { attr_name : [v1, v2, v3, ...] }
    """

    def __init__(self, data: dict[str, list[float]]):
        self.data = data

    # ---------- factory ----------
    @classmethod
    def from_file(cls, path: str):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in [".txt", ".csv"]:
            data = cls._load_text_like(path)
        elif suffix == ".json":
            data = cls._load_json(path)
        elif suffix in [".xlsx", ".xls"]:
            data = cls._load_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return cls(data)

    # ---------- loaders ----------
    @staticmethod
    def _load_text_like(path: Path) -> dict[str, list[float]]:
        df = pd.read_csv(path, header=None)
        df = df.apply(pd.to_numeric, errors="ignore")

        data = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0])
            values = row.iloc[1:].dropna().astype(float).tolist()
            data[key] = values

        return data

    @staticmethod
    def _load_json(path: Path) -> dict[str, list[float]]:
        with open(path, "r") as f:
            data = json.load(f)

        # basic validation
        if not all(isinstance(v, list) for v in data.values()):
            raise ValueError("JSON values must be lists")

        return data

    @staticmethod
    def _load_excel(path: Path) -> dict[str, list[float]]:
        df = pd.read_excel(path, header=None)
        df = df.apply(pd.to_numeric, errors="ignore")

        data = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0])
            values = row.iloc[1:].dropna().astype(float).tolist()
            data[key] = values

        return data

    # ---------- plotting ----------
    def plot(
        self,
        title: str | None = None,
        xlabel: str = "Index",
        ylabel: str = "Value",
        save: str | None = None,
        figsize=(10, 6),
    ):
        plt.figure(figsize=figsize)

        # modern color palette (20 colors)
        colors = plt.get_cmap("tab20").colors

        for i, (attr, values) in enumerate(self.data.items()):
            x = range(1, len(values) + 1)
            plt.plot(
                x,
                values,
                label=attr,
                color=colors[i % len(colors)],
                linewidth=2,
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300)
        else:
            plt.show()
