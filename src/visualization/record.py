from typing import Any, Dict, List, Callable, Iterable
import json


class RecordBuffer:
    def __init__(self):
        self._data: Dict[str, List[Any]] = {}

    # ---------- 基本协议 ----------
    def __contains__(self, attr: str) -> bool:
        return attr in self._data

    def __getitem__(self, attr: str) -> List[Any]:
        return self._data[attr]

    def keys(self):
        return self._data.keys()

    def lens(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._data.items()}

    @property
    def data(self) -> Dict[str, List[Any]]:
        return self._data

    # ---------- 写入 ----------
    def append(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            self._data.setdefault(k, []).append(v)

    # ---------- 删除 / 过滤 ----------
    def drop_attr(self, attr: str) -> None:
        self._data.pop(attr, None)

    def filter(self, attr: str, predicate: Callable[[Any], bool]) -> None:
        self._data[attr] = [v for v in self._data[attr] if predicate(v)]

    # ---------- 导出（列式） ----------
    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(self._data, f, indent=indent)

    def to_txt(self, path: str, sep: str = ", ") -> None:
        with open(path, "w") as f:
            for k, v in self._data.items():
                f.write(f"{k}: {sep.join(map(str, v))}\n")
