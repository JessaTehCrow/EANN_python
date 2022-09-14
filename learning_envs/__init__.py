import os, importlib

from util import root_directory

_dirlist = os.listdir(root_directory+"/learning_envs")
_folders = [item for item in _dirlist if not "_" in item and os.path.isdir(root_directory+f"/learning_envs/{item}")]

environments = {}

# Load environments
for _env in _folders:
    try:
        lib = importlib.import_module(f"learning_envs.{_env}")
    except Exception as e:
        print(f"Unable to initialize environment {_env}: {e}")
        continue
    environments[_env] = {
        "module" : lib,
        "path"   : root_directory+f"/learning_envs/{_env}"
    }