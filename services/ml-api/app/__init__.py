"""FastAPI service exposing the visual speech recognition pipeline.

The service directory name contains a hyphen and so cannot itself be a Python
package. Rather than require an editable install just to run the server, the
repository root is put on ``sys.path`` here — the single place it happens.

Run from ``services/ml-api``:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
