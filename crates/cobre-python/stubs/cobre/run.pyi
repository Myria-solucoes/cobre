from pathlib import Path
from typing import Any, Mapping, Optional, Union

def run(
    case_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    threads: Optional[int] = None,
    skip_simulation: Optional[bool] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]: ...
