"""Runtime path configuration for local and Google Colab executions."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Dict, Optional, Union


RUNTIME_ENV = "KBPROJECTION_RUNTIME"
PROJECT_ROOT_ENV = "KBPROJECTION_PROJECT_ROOT"
DATA_DIR_ENV = "KBPROJECTION_DATA_DIR"
CACHE_ROOT_ENV = "KBPROJECTION_CACHE_ROOT"
RESULTS_DIR_ENV = "KBPROJECTION_RESULTS_DIR"
CACHE_DIR_ENV = "KBPROJECTION_THIRD_PARTY_CACHE"


@dataclass(frozen=True)
class RuntimePaths:
    runtime: str
    project_root: Path
    data_dir: Path
    cache_root: Path
    results_dir: Path
    third_party_cache: Path
    hf_home: Path
    hf_datasets_cache: Path
    transformers_cache: Path
    sentence_transformers_home: Path
    nltk_data: Path
    torch_home: Path

    def as_dict(self) -> Dict[str, Path]:
        values = asdict(self)
        values.pop("runtime", None)
        return values


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def detect_runtime(explicit: Optional[str] = None) -> str:
    runtime = explicit or os.environ.get(RUNTIME_ENV)
    if runtime:
        runtime = runtime.strip().lower()
    else:
        runtime = "colab" if is_colab() else "local"

    if runtime not in {"local", "colab"}:
        raise ValueError(
            f"Unsupported {RUNTIME_ENV}={runtime!r}. Expected 'local' or 'colab'."
        )
    return runtime


def mount_google_drive(
    mount_point: Union[str, Path] = "/content/drive",
    force_remount: bool = False,
) -> bool:
    if not is_colab():
        return False

    from google.colab import drive  # type: ignore

    drive.mount(str(mount_point), force_remount=force_remount)
    return True


def default_project_root(runtime: Optional[str] = None) -> Path:
    selected_runtime = detect_runtime(runtime)
    override = os.environ.get(PROJECT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()

    if selected_runtime == "colab":
        return Path("/content/drive/MyDrive/kbprojection")

    return Path.cwd().resolve()


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser().resolve()
    return default


def configure_runtime(
    project_root: Optional[Union[str, Path]] = None,
    runtime: Optional[str] = None,
    mount_drive: Optional[bool] = None,
    drive_mount_point: Union[str, Path] = "/content/drive",
    force_remount: bool = False,
    create_dirs: bool = True,
    set_env: bool = True,
) -> RuntimePaths:
    selected_runtime = detect_runtime(runtime)

    should_mount = selected_runtime == "colab" if mount_drive is None else mount_drive
    if should_mount:
        mount_google_drive(drive_mount_point, force_remount=force_remount)

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else default_project_root(selected_runtime)
    )

    data_dir = _path_from_env(DATA_DIR_ENV, root / "data")
    cache_root = _path_from_env(CACHE_ROOT_ENV, root / "experiment_cache")
    results_dir = _path_from_env(RESULTS_DIR_ENV, root / "experiment_results")
    third_party_cache = _path_from_env(CACHE_DIR_ENV, root / ".cache")

    paths = RuntimePaths(
        runtime=selected_runtime,
        project_root=root,
        data_dir=data_dir,
        cache_root=cache_root,
        results_dir=results_dir,
        third_party_cache=third_party_cache,
        hf_home=third_party_cache / "huggingface",
        hf_datasets_cache=third_party_cache / "huggingface" / "datasets",
        transformers_cache=third_party_cache / "huggingface" / "transformers",
        sentence_transformers_home=third_party_cache / "sentence_transformers",
        nltk_data=third_party_cache / "nltk",
        torch_home=third_party_cache / "torch",
    )

    if create_dirs:
        for path in paths.as_dict().values():
            path.mkdir(parents=True, exist_ok=True)

    if set_env:
        os.environ[RUNTIME_ENV] = selected_runtime
        os.environ[PROJECT_ROOT_ENV] = str(paths.project_root)
        os.environ[DATA_DIR_ENV] = str(paths.data_dir)
        os.environ[CACHE_ROOT_ENV] = str(paths.cache_root)
        os.environ[RESULTS_DIR_ENV] = str(paths.results_dir)
        os.environ[CACHE_DIR_ENV] = str(paths.third_party_cache)
        os.environ["HF_HOME"] = str(paths.hf_home)
        os.environ["HF_DATASETS_CACHE"] = str(paths.hf_datasets_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(paths.transformers_cache)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(paths.sentence_transformers_home)
        os.environ["NLTK_DATA"] = str(paths.nltk_data)
        os.environ["TORCH_HOME"] = str(paths.torch_home)

    return paths
