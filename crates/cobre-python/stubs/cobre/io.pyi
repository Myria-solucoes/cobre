from pathlib import Path
from typing import Any, Union

from . import model as model

def load_case(path: Union[str, Path]) -> model.System: ...
def validate(path: Union[str, Path]) -> dict[str, Any]: ...
