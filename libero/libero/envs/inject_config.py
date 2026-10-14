import json
import os


def _find_industry_config(bddl_file_name):
    """Find the nearest repository-level Industry configuration file."""
    current_dir = os.path.abspath(os.path.dirname(bddl_file_name))
    while True:
        candidate = os.path.join(current_dir, "benchmark", "industry_env_args.json")
        if os.path.isfile(candidate):
            return candidate
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def inject_industry_config(problem_name, bddl_file_name, kwargs):
    """Fill missing Industry kwargs from the configuration for this BDDL.

    Explicit caller values are authoritative.  In particular, a staged
    benchmark manifest passes ``set_scene`` and must not be silently replaced
    by a live JSON configuration with the same task name.
    """
    is_industry = "industry" in str(problem_name).lower()
    task_name = os.path.basename(bddl_file_name).replace(".bddl", "")
    json_path = _find_industry_config(bddl_file_name)
    if json_path is None:
        if is_industry:
            raise SyntaxError(
                "Missing industry_env_args.json for Industry task '{}'".format(task_name)
            )
        return

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            configurations = json.load(handle)
        case_insensitive = {key.lower(): value for key, value in configurations.items()}
        task_config = case_insensitive.get(task_name.lower())
        if task_config is None:
            if is_industry:
                raise SyntaxError(
                    "Missing industry_env_args.json entry for Industry task '{}'".format(
                        task_name
                    )
                )
            return

        if is_industry:
            configured_takt = task_config.get("takt_steps")
            if (
                isinstance(configured_takt, bool)
                or not isinstance(configured_takt, int)
                or configured_takt <= 0
            ):
                raise SyntaxError(
                    "Invalid takt_steps for Industry task '{}' in '{}'".format(
                        task_name, json_path
                    )
                )
            supplied_takt = kwargs.get("takt_steps")
            if supplied_takt is not None and int(supplied_takt) != configured_takt:
                raise SyntaxError(
                    "Industry takt_steps mismatch for '{}': caller={}, config={}".format(
                        task_name, supplied_takt, configured_takt
                    )
                )
            kwargs["takt_steps"] = configured_takt

        if "set_scene" in task_config and kwargs.get("set_scene") is None:
            kwargs["set_scene"] = task_config["set_scene"]

        configured_env_args = task_config.get("env_args", {})
        existing_env_args = dict(kwargs.get("env_args", {}))
        additions = {
            key: value
            for key, value in configured_env_args.items()
            if key not in existing_env_args and key not in kwargs
        }
        if additions:
            existing_env_args.update(additions)
            kwargs["env_args"] = existing_env_args

        if not is_industry and "takt_steps" in task_config and "takt_steps" not in kwargs:
            kwargs["takt_steps"] = task_config["takt_steps"]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SyntaxError(
            "Could not inject Industry configuration for '{}': {}".format(task_name, error)
        ) from error
