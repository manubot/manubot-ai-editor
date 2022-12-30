import json
import os
from pathlib import Path

from manubot.ai_editor import env_vars
from manubot.ai_editor.models import ManuscriptRevisionModel
from manubot.ai_editor.utils import get_yaml_field, SENTENCE_END_PATTERN


class ManuscriptEditor:
    """
    It provides functions to revise the Markdown files (*.md) of a Manubot-based manuscript.
    The only mandatory requirement is the path to the "content" directory of the manuscript.

    Args:
        content_dir: Path to the "content" directory of a Manubot-based manuscript.
    """

    def __init__(self, content_dir: str | Path):
        self.content_dir = Path(content_dir)

        metadata_file = self.content_dir / "metadata.yaml"
        assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"
        self.title = get_yaml_field(metadata_file, "title")
        self.keywords = get_yaml_field(metadata_file, "keywords")

    @staticmethod
    def revise_and_write_paragraph(
        paragraph: list[str],
        section_name: str,
        revision_model: ManuscriptRevisionModel,
        outfile=None,
    ) -> None | tuple[str]:
        """
        Revises and writes a paragraph to the output file.

        Arguments:
            paragraph: list of lines of the paragraph.
            section_name: name of the section the paragraph belongs to.
            revision_model: model to use for revision.
            outfile: file object to write the revised paragraph to.

        Returns:
            None if outfile is specified. Otherwise, it returns a tuple with
            the submitted paragraph and the revised paragraph.
        """
        # Process the paragraph and revise it with model
        paragraph_text = " ".join(paragraph)

        error_message = None
        try:
            paragraph_revised = revision_model.revise_paragraph(
                paragraph_text,
                section_name,
            )

            if paragraph_revised.strip() == "":
                raise Exception("The AI model returned an empty string ('')")
        except Exception as e:
            error_message = f"""
<!--
ERROR: this paragraph could not be revised with the AI model due to the following error:

{str(e)}
-->
            """.strip()

        if error_message is not None:
            paragraph_revised = (
                error_message
                + "\n"
                + ManuscriptEditor.convert_sentence_ends_to_newlines(paragraph_text)
            )
        else:
            # put sentences into new lines
            paragraph_revised = ManuscriptEditor.convert_sentence_ends_to_newlines(
                paragraph_revised
            )

        if outfile is not None:
            outfile.write(paragraph_revised + "\n")
        else:
            return paragraph_text, paragraph_revised

    @staticmethod
    def convert_sentence_ends_to_newlines(paragraph: str) -> str:
        """
        Converts sentence ends to newlines.

        Args:
            paragraph: paragraph to convert.

        Returns:
            Converted paragraph.
        """
        return SENTENCE_END_PATTERN.sub(r".\n\1", paragraph)

    @staticmethod
    def get_section_from_filename(filename: str) -> str | None:
        """
        Returns the section name of a file based on its filename.
        """
        filename = filename.lower()

        if env_vars.SECTIONS_MAPPING in os.environ:
            sections_mapping = os.environ[env_vars.SECTIONS_MAPPING]
            try:
                sections_mapping = json.loads(sections_mapping)
                if filename in sections_mapping:
                    return sections_mapping[filename]
            except json.JSONDecodeError:
                print(
                    f"Invalid JSON in environment variable {env_vars.SECTIONS_MAPPING}",
                    flush=True,
                )

        if "abstract" in filename:
            return "abstract"
        elif "introduction" in filename:
            return "introduction"
        elif "result" in filename:
            return "results"
        elif "discussion" in filename:
            return "discussion"
        elif "conclusion" in filename:
            return "conclusions"
        elif "method" in filename:
            return "methods"
        elif "supplementary" in filename:
            return "supplementary_material"
        else:
            return None

    @staticmethod
    def line_is_not_part_of_paragraph(line: str, include_blank=True) -> bool:
        prefixes = ("![", "|", "<!--", "$$", "#", "```")
        return line.startswith(prefixes) or (include_blank and line.strip() == "")

    def revise_file(
        self,
        input_filename: str,
        output_dir: Path | str,
        revision_model: ManuscriptRevisionModel,
        section_name: str = None,
    ):
        """
        It revises an entire Markdown file and writes the revised file to the output directory.
        The output file will have the same name as the input file.

        Args:
            input_filename (str): name of the file to revise. It must exists in the content directory of the manuscript.
            output_dir (Path | str): path to the directory where the revised file will be written.
            revision_model (ManuscriptRevisionModel): model to use for revision.
            section_name (str, optional): Defaults to None. If so, it will be inferred from the filename.
        """
        input_filepath = self.content_dir / input_filename
        assert input_filepath.exists(), f"Input file {input_filepath} does not exist"

        output_dir = Path(output_dir).resolve()
        assert output_dir.exists(), f"Output directory {output_dir} does not exist"
        output_filepath = output_dir / input_filename

        # infer section name from input filename if not provided
        if section_name is None:
            section_name = self.get_section_from_filename(input_filename)

        with open(input_filepath, "r") as infile, open(output_filepath, "w") as outfile:
            # Initialize a temporary list to store the lines of the current paragraph
            paragraph = []

            for line in infile:
                # if line is starting either an "image paragraph", a "table paragraph" or a "html comment paragraph",
                # then skip all lines until the end of that paragraph
                if self.line_is_not_part_of_paragraph(line, include_blank=False):
                    while line is not None and line.strip() != "":
                        outfile.write(line)
                        line = next(infile, None)

                # stop if we reached the end of the file
                if line is None:
                    break

                # if the line is empty and we didn't start a paragraph yet,
                # write it directly to the output file
                if line.strip() == "" and len(paragraph) == 0:
                    outfile.write(line)

                # If the line is blank, it indicates the end of a paragraph
                elif line.strip() == "":
                    # revise and write paragraph to output file
                    self.revise_and_write_paragraph(
                        paragraph, section_name, revision_model, outfile
                    )

                    # and also the current line, which is the end of the
                    # paragraph
                    outfile.write(line)

                    # clear the paragraph list
                    paragraph = []

                # Otherwise, add the line to the paragraph list
                else:
                    paragraph.append(line.strip())

            # If there's any remaining paragraph, process and write it to the
            # output file
            if paragraph:
                self.revise_and_write_paragraph(
                    paragraph, section_name, revision_model, outfile
                )

    def revise_manuscript(
        self,
        output_dir: Path | str,
        revision_model: ManuscriptRevisionModel,
        debug: bool = False,
    ):
        """
        Revises all the files in the content directory of the manuscript sorted
        by name, and writes each file in the output directory.
        """
        for filename in sorted(self.content_dir.glob("*.md")):
            filename_section = self.get_section_from_filename(filename.name)
            if filename_section is None:
                continue

            if debug:
                print(f"Revising {filename.name}", flush=True)

            self.revise_file(
                filename.name,
                output_dir,
                revision_model,
                section_name=filename_section,
            )
