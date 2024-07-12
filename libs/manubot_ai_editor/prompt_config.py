"""
Implements reading filename->prompt resolution rules from a YAML file, via the
class ManuscriptPromptConfig.
"""

import os
import re
from typing import Optional
import yaml
from pathlib import Path

from manubot_ai_editor.utils import get_obj_path

# if returned as the prompt from get_prompt_for_filename() then the file should
# be ignored
IGNORE_FILE = "__IGNORE_FILE__"


class ManuscriptConfigException(Exception):
    """
    Parent class for exceptions raised by ManuscriptConfig's loading process.
    """

    pass


class ManuscriptPromptConfig:
    """
    Loads configuration from two YAML files in 'config_dir':
    -  ai-revision-prompts.yaml, which contains custom prompt definitions and/or
    mappings of prompts to files
    - ai-revision-config.yaml, containing general configuration for the AI
    revision process

    After loading, the main use of this class is to resolve a prompt for a given
    filename. This is done by calling config.get_prompt_for_filename(<filename>),
    which uses both the 'ai-revision-prompts.yaml' and 'ai-revision-config.yaml'
    files to determine the prompt for a given filename.
    """

    def __init__(self, config_dir: str | Path, title: str, keywords: str) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else None
        self.config = self._load_config()
        self.prompts, self.prompts_files = self._load_custom_prompts()

        # validation: both self.config.files.matchings and self.prompts_files
        # specify filename-to-prompt mappings; if both are present, we use
        # self.config.files, but warn the user that they should only use one
        if (
            self.prompts_files is not None and
            self.config is not None and
            self.config.get('files', {}).get('matchings') is not None
        ):
            print(
                "WARNING: Both 'ai-revision-config.yaml' and 'ai-revision-prompts.yaml' specify filename-to-prompt mappings. "
                "Only the 'ai-revision-config.yaml' file's file.matchings section will be used; prompts_files will be ignored."
            )

        # storing these so they can be interpolated into prompts
        self.title = title
        self.keywords = keywords

    def _load_config(self) -> dict:
        """
        Loads general configuration from ai-revision-config.yaml
        """

        # if no config folder was specified, we just resort to the default of
        # not using the custom prompts system at all
        if self.config_dir is None:
            return None

        config_file_path = os.path.join(self.config_dir, "ai-revision-config.yaml")

        try:
            with open(config_file_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return None

    def _load_custom_prompts(self) -> tuple[dict, dict]:
        """
        Loads custom prompts from ai-revision-prompts.yaml. The file
        must contain either 'prompts' or 'prompts_files' as top-level keys.

        'prompts' is a dictionary where keys are filenames and values are
        prompts. For example: '{"intro": "proofread the following paragraph"}'.
        The key can be used in the configuration file to specify a prompt for
        a given file.

        """

        # same as _load_config, if no config folder was specified, we just
        if self.config_dir is None:
            return (None, None)
        
        prompt_file_path = os.path.join(self.config_dir, "ai-revision-prompts.yaml")

        try:
            with open(prompt_file_path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            # if the file doesn't exist, return None for both prompts and prompts_files
            return (None, None)

        # validate the existence of at least one of the keys we require
        if "prompts" not in data and "prompts_files" not in data:
            raise ManuscriptConfigException(
                'The "ai-revision-prompts.yaml" YAML file must contain a "prompts" or a "prompts_files" key.'
            )

        # if the top-level key was 'prompts', that implies we need the `ai-revision-config.yaml`
        # file to match those prompts to filenames, so raise an exception if it doesn't exist
        if "prompts" in data and not self.config:
            raise ManuscriptConfigException(
                'The "ai-revision-config.yaml" YAML file must exist if "ai-revision-prompts.yaml" begins with the "prompts" key.'
            )

        prompts = data.get("prompts")
        prompts_files = data.get("prompts_files")

        return (prompts, prompts_files)

    def get_prompt_for_filename(
        self, filename: str, use_default: bool = True
    ) -> tuple[Optional[str], Optional[re.Match]]:
        """
        Retrieves the prompt for a given filename. It checks the following sources
        for a match in order:
        - the 'ignore' list in ai-revision-config.yaml; if matched, returns None.
        - the 'matchings' list in ai-revision-config.yaml; if matched, returns
        the value for the referenced prompt, specified in ai-revision-prompts.yaml,
        - the 'prompts_files' collection in ai-revision-prompts.yaml; if matched,
        returns the prompt specified alongside the file pattern from that file.

        If a match is found, returns a tuple of the prompt text and the match object.
        If the file is in the ignore list, returns (None, m), where m is the match
        object that matched the ignore pattern.
        If nothing matched and 'use_default' is True, returns (default_prompt,
        None) where 'default_prompt' is the default prompt specified in
        ai-revision-config.yaml, if available.
        """

        # first, check the ignore list to see if we should bail early
        for ignore in get_obj_path(self.config, ("files", "ignore"), missing=[]):
            if m := re.search(ignore, filename):
                return (IGNORE_FILE, m)

        # if both ai-revision-config.yaml specifies files.matchings and
        # ai-revision-prompts.yaml specifies prompts_files, then files.matchings
        # takes precedence.
        # (the user is notified of this in a validation warning in __init__)
        
        # then, consult ai-revision-config.yaml's 'matchings' collection if a
        # match is found, use the prompt ai-revision-prompts.yaml
        for entry in get_obj_path(self.config, ("files", "matchings"), missing=[]):
            # iterate through all the 'matchings' entries, trying to find one
            # that matches the current filename
            for pattern in entry["files"]:
                if m := re.search(pattern, filename):
                    # since we matched, use the 'prompts' collection to return a
                    # named prompt corresponding to the one from the 'matchings'
                    # collection
                    resolved_prompt = None

                    if self.prompts:
                        resolved_prompt = self.prompts.get(entry["prompt"], None)

                        if resolved_prompt is not None:
                            resolved_prompt = resolved_prompt.strip()

                    return ( resolved_prompt, m, )

        # since we haven't found a match yet, consult ai-revision-prompts.yaml's
        # 'prompts_files' collection
        if self.prompts_files:
            for pattern, prompt in self.prompts_files.items():
                if m := re.search(pattern, filename):
                    return (prompt.strip() if prompt is not None else IGNORE_FILE, m)

        # finally, resolve the default prompt, which we do by:
        # 1) checking if the 'default_prompt' key exists in the config file, using 'default' if it's unspecified
        # 2) use whatever we resolved to reference the prompt from the 'prompts' collection
        # 3) if we can't resolve a default prompt for whatever reason, return None
        resolved_default_prompt = None
        if use_default and self.prompts is not None:
            resolved_default_prompt = self.prompts.get(
                get_obj_path(self.config, ("files", "default_prompt")),
                None
            )

            if resolved_default_prompt is not None:
                resolved_default_prompt = resolved_default_prompt.strip()
        
        return (resolved_default_prompt, None)
