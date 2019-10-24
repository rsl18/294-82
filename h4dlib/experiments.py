"""
h4dlib.experiments is a module to help manage experiments.
"""
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from getpass import getuser
from pathlib import Path, PurePath
from typing import Any, Dict

import git
import numpy as np
import setproctitle

from h4dlib.config import h4dconfig


@dataclass
class GitInfo:
    """Simple data class to hold some info about the git repo"""
    branch: str
    commit_hash: str
    is_dirty: bool


# Calculates output_dir and logs all input args
class ExperimentManager:
    """This class manages info and operations related to tracking experiments"""

    def __init__(self, framework_name: str, configs: Dict[str, Any] = None):
        self.base_path: Path = h4dconfig.EXPERIMENTS_DIR
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.output_dir: Path = self.base_path / framework_name / getuser() / timestamp
        self.log_dir = self.output_dir / "logs"
        self.diff_dir = self.output_dir / "diffs"

        # Process title:
        title = f"{sys.argv[0]}-{self.output_dir}".replace(str(self.base_path), "")
        print("setting process title to: ", title)
        setproctitle.setproctitle(title)

        # Create folders:
        folders = [self.output_dir, self.log_dir, self.diff_dir]
        for folder in folders:
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

        if configs is not None:
            self.start(configs)

    def start(self, configs: Dict[str, Any]) -> None:
        """
        Starts the experiment by logging configs argument to json file. You don't need
        to call this if you passed in configs into the ExperimentManager constructor
        """
        # Save experiment config to json:
        configs["git_info"] = self._get_repo_info()
        session_name = self.output_dir / "experiment_desc.json"
        with open(session_name, "w") as session_config:
            json.dump(
                configs, session_config, cls=CustomJSONEncoder, sort_keys=True, indent=4
            )

    def _get_repo_info(self):
        """
        Returns GitInfo instance with info about branch, commit hash, and is_dirty
        state. If git repo is dirty, saves git diff and untracked files into
        self.diff_dir.
        """
        repo: git.Repo = git.Repo(h4dconfig.ROOT_DIR)
        info = GitInfo(
            repo.active_branch.name, repo.active_branch.commit.hexsha, repo.is_dirty()
        )
        if info.is_dirty:
            repo.git.reset()
            with open(self.diff_dir / "working_dir.diff", "w") as diff:
                diff.write(repo.git.diff())
            for file in repo.untracked_files:
                dest_path = (self.diff_dir / "untracked" / file).parent
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(h4dconfig.ROOT_DIR / file, dest_path)
        return info

    def get_output_dir(self) -> Path:
        """
        Return the path to the output dir of this experiment.
        """
        return self.output_dir

    def mark_dir_if_complete(self) -> None:
        """
        Append C to front of output dir name if training finishes successfully.
        """
        output_dir_name = self.get_output_dir().absolute()
        last_dir_name = output_dir_name.parts[-1]
        new_dir_name = 'C' + last_dir_name
        path_to_parent = output_dir_name.parents[0]
        new_dir_path = PurePath(path_to_parent, new_dir_name)
        os.rename(output_dir_name, new_dir_path)
        self.output_dir = new_dir_path

class CustomJSONEncoder(json.JSONEncoder):
    """
    This class is used by json.dump() to help serilize numpy arrays and other types
    that aren't serializable.
    """

    def default(self, o):   # pylint: disable=E0202
        if type(o).__module__ == np.__name__:
            if isinstance(o, np.ndarray):
                return o.tolist()
            else:
                return o.item()
        elif is_dataclass(o):
            return asdict(o)
        raise TypeError("Unknown type:", type(o))
