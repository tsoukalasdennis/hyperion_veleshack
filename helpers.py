import os

import httpx

IDE_BACKEND_URL = os.environ.get("IDE_BACKEND_URL", "http://localhost:3001/api")


class ReadFileError(Exception):
    """The IDE could not give us the file."""


class ValidateFileError(Exception):
    """The IDE could not validate the file."""


async def read_file(path: str) -> str:
    """Return the contents of a file in the IDE workspace.

    `path` is either a full path (`demo/app.yaml`) or just a file name, in which
    case the IDE searches the whole workspace for it. Raises ReadFileError with a
    message you can hand straight to a model.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{IDE_BACKEND_URL}/agent/file", params={"path": path}
            )
    except httpx.HTTPError as exc:
        raise ReadFileError(
            f"cannot reach the IDE backend at {IDE_BACKEND_URL} ({exc})"
        ) from exc

    if response.status_code == 404:
        raise ReadFileError(f"{path} is not in the workspace")

    if response.status_code == 409:
        matches = ", ".join(response.json().get("matches", []))
        raise ReadFileError(
            f"several files are named {path} ({matches}) - use the full path"
        )

    if response.status_code != 200:
        raise ReadFileError(
            response.json().get("error", f"HTTP {response.status_code}")
        )

    return response.json().get("content", "")


async def validate_file(path: str) -> dict:
    """Validate a file in the IDE workspace and return the report.

    `path` is either a full path (`demo/app.yaml`) or just a file name, in which
    case the IDE searches the whole workspace for it. The report is a dict with
    `path`, `type`, `valid`, `errors` and `warnings`. Raises ValidateFileError
    with a message you can hand straight to a model.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{IDE_BACKEND_URL}/agent/validation/file", params={"path": path}
            )
    except httpx.HTTPError as exc:
        raise ValidateFileError(
            f"cannot reach the IDE backend at {IDE_BACKEND_URL} ({exc})"
        ) from exc

    if response.status_code == 404:
        raise ValidateFileError(f"{path} is not in the workspace")

    if response.status_code == 409:
        matches = ", ".join(response.json().get("matches", []))
        raise ValidateFileError(
            f"several files are named {path} ({matches}) - use the full path"
        )

    if response.status_code != 200:
        raise ValidateFileError(
            response.json().get("error", f"HTTP {response.status_code}")
        )

    return response.json()

class ResolvePathError(Exception):
    """The workspace path could not be resolved."""


async def list_workspace_files(path: str = "") -> list[str]:
    """Recursively list files in the IDE workspace."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{IDE_BACKEND_URL}/files",
                params={"path": path},
            )
    except httpx.HTTPError as exc:
        raise ResolvePathError(
            f"cannot reach the IDE backend at {IDE_BACKEND_URL} ({exc})"
        ) from exc

    if response.status_code != 200:
        raise ResolvePathError(
            response.json().get("error", f"HTTP {response.status_code}")
        )

    files = []

    for item in response.json():
        if item["type"] == "file":
            files.append(item["path"])
        elif item["type"] == "folder":
            files.extend(await list_workspace_files(item["path"]))

    return files


async def resolve_existing_path(path: str) -> str:
    """Resolve a user-provided path against the actual workspace."""
    requested = path.strip().lstrip("/")

    if not requested:
        raise ResolvePathError("No file path was provided.")

    workspace_files = await list_workspace_files()

    # 1. Exact path
    exact_matches = [
        file_path
        for file_path in workspace_files
        if file_path == requested
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    # 2. Exact basename
    basename_matches = [
        file_path
        for file_path in workspace_files
        if file_path.rsplit("/", 1)[-1] == requested
    ]

    if len(basename_matches) == 1:
        return basename_matches[0]

    if len(basename_matches) > 1:
        matches = ", ".join(basename_matches)
        raise ResolvePathError(
            f"Several files match '{requested}': {matches}. "
            "Please specify the full path."
        )

    # 3. Stem match, e.g. "myfile" -> "myfile.yaml"
    stem_matches = [
        file_path
        for file_path in workspace_files
        if file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0] == requested
    ]

    if len(stem_matches) == 1:
        return stem_matches[0]

    if len(stem_matches) > 1:
        matches = ", ".join(stem_matches)
        raise ResolvePathError(
            f"Several files match '{requested}': {matches}. "
            "Please specify the full path."
        )

    raise ResolvePathError(
        f"I couldn't find a file matching '{requested}' in the workspace."
    )
