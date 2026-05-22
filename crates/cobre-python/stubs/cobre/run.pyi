from pathlib import Path
from typing import Any, Optional, Union

def run(
    case_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    threads: Optional[int] = None,
    skip_simulation: Optional[bool] = None,
) -> dict[str, Any]: ...
