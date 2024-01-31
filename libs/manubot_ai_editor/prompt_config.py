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
    Loads configuration from two YAML files in 'content_dir':
    -  ai_revision-prompts.yaml, which contains custom prompt definitions and/or
    mappings of prompts to files
    - ai_revision-config.yaml, containing general
    configuration for the AI revision process

    After loading, the main use of this class is to resolve a prompt for a given
    filename. This is done by calling config.get_prompt_for_filename(<filename>),
    which uses both the 'ai_revision-prompts.yaml' and 'ai_revision-config.yaml'
    files to determine the prompt for a given filename.
    """
    def __init__(self, content_dir: str, title: str, keywords: str) -> None:
        self.content_dir = Path(content_dir)
        self.config = self._load_config()
        self.prompts, self.prompts_files = self._load_custom_prompts()

        # storing these so they can be interpolated into prompts
        self.title = title
        self.keywords = keywords

    def _load_config(self) -> dict:
        """
        Loads general configuration from ai_revision-config.yaml
        """
        
        config_file_path = os.path.join(self.content_dir, "ai_revision-config.yaml")

        try:
            with open(config_file_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return None

        
    def _load_custom_prompts(self) -> (dict, dict):
        """
        Loads custom prompts from ai_revision-prompts.yaml. The file
        must contain either 'prompts' or 'prompts_files' as top-level keys.

        'prompts' is a dictionary where keys are filenames and values are
        prompts. For example: '{"intro": "proofread the following paragraph"}'.
        The key can be used in the configuration file to specify a prompt for
        a given file.

        """

        prompt_file_path = os.path.join(self.content_dir, "ai_revision-prompts.yaml")

        try:
            with open(prompt_file_path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            # if the file doesn't exist, return None for both prompts and prompts_files
            return (None, None)

        # validate the existence of at least one of the keys we require
        if 'prompts' not in data and 'prompts_files' not in data:
            raise ManuscriptConfigException('The "ai_revision-prompts.yaml" YAML file must contain a "prompts" or a "prompts_files" key.')

        # if the top-level key was 'prompts', that implies we need the `ai_revision-config.yaml`
        # file to match those prompts to filenames, so raise an exception if it doesn't exist
        if 'prompts' in data and not self.config:
            raise ManuscriptConfigException(
                'The "ai_revision-config.yaml" YAML file must exist if "ai_revision-prompts.yaml" begins with the "prompts" key.'
            )

        prompts = data.get('prompts')
        prompts_files = data.get('prompts_files')

        return (prompts, prompts_files)

    def get_prompt_for_filename(self, filename: str, use_default: bool = True) -> (Optional[str], Optional[re.Match]):
        """
        Retrieves the prompt for a given filename. It checks the following sources
        for a match in order:
        - the 'ignore' list in ai_revision-config.yaml; if matched, returns None.
        - the 'matchings' list in ai_revision-config.yaml; if matched, returns
        the value for the referenced prompt, specified in ai_revision-prompts.yaml,
        - the 'prompts_files' collection in ai_revision-prompts.yaml; if matched,
        returns the prompt specified alongside the file pattern from that file.

        If a match is found, returns a tuple of the prompt text and the match object.
        If the file is in the ignore list, returns (None, m), where m is the match
        object that matched the ignore pattern.
        If nothing matched and 'use_default' is True, returns (default_prompt,
        None) where 'default_prompt' is the default prompt specified in
        ai_revision-config.yaml, if available.
        """

        # first, check the ignore list to see if we should bail early
        for ignore in get_obj_path(self.config, ('files', 'ignore'), missing=[]):
            if (m := re.search(ignore, filename)):
                return (IGNORE_FILE, m)

        # FIXME: which takes priority, the files collection in ai_revision-config.yaml
        #  or the prompt_file? we went with config taking precendence for now

        # then, consult ai_revision-config.yaml's 'matchings' collection if a
        # match is found, use the prompt ai_revision-prompts.yaml
        for entry in get_obj_path(self.config, ('files', 'matchings'), missing=[]):
            # iterate through all the 'matchings' entries, trying to find one
            # that matches the current filename
            for pattern in entry['files']:
                if (m := re.search(pattern, filename)):
                    # since we matched, use the 'prompts' collection to return a
                    # named prompt corresponding to the one from the 'matchings'
                    # collection
                    return (
                        self.prompts.get(entry['prompt'], None) if self.prompts else None, m
                    )

        # since we haven't found a match yet, consult ai_revision-prompts.yaml's
        # 'prompts_files' collection
        if self.prompts_files:
            for pattern, prompt in self.prompts_files.items():
                if (m := re.search(pattern, filename)):
                    return (prompt if prompt is not None else IGNORE_FILE, m)

        # finally, return the default prompt
        return (
            get_obj_path(self.config, ('files', 'default_prompt')) if use_default else None,
            None
        )
