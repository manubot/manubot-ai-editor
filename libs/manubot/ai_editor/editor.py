import json
import os
from pathlib import Path

from manubot.ai_editor import env_vars
from manubot.ai_editor.models import ManuscriptRevisionModel
from manubot.ai_editor.utils import (
    get_yaml_field,
    SENTENCE_END_PATTERN,
    SIMPLE_SENTENCE_END_PATTERN,
)


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
    def prepare_paragraph(paragraph: list[str]) -> str:
        """
        It takes a list of sentences that are part of a paragraph and joins them
        into a single string. The paragraph might have Equations, which are
        between "$$" and have an identifier between curly brackets. There are
        two kinds of paragraphs:
        1) "Simple paragraphs" are a set of sentences with no Equations;
        2) "Equation paragraphs" are a set of sentences where at least one
        sentence has an Equation. When joining sentences with Equations, a
        newline is added before and after the Equation.
        """
        paragraph_text = ""

        paragraph = iter(paragraph)
        for sentence in paragraph:
            sentence = sentence.strip()

            if sentence == "":
                paragraph_text += "\n"
            elif sentence.startswith("$$"):
                # this is an equation
                equation_sentences = [sentence]

                sentence = next(paragraph, None)
                while sentence is not None and not sentence.startswith("$$"):
                    equation_sentences.append(sentence)
                    sentence = next(paragraph, None)

                if sentence is not None:
                    equation_sentences.append(sentence)

                paragraph_text += "\n".join(equation_sentences) + "\n"
            else:
                simple_sentences = [sentence]

                sentence = next(paragraph, None)
                while sentence is not None and sentence != "":
                    simple_sentences.append(sentence)
                    sentence = next(paragraph, None)

                suffix = "\n"
                if sentence == "":
                    suffix += "\n"

                paragraph_text += " ".join(simple_sentences) + suffix

        return paragraph_text

    @staticmethod
    def revise_and_write_paragraph(
        paragraph: list[str],
        section_name: str,
        revision_model: ManuscriptRevisionModel,
        outfile=None,
    ) -> None | tuple[str, str]:
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
        paragraph_text = ManuscriptEditor.prepare_paragraph(paragraph)

        # revise paragraph only if it has all these properties: 1) it has at
        # least two sentences, 2) it has in total at least 60 words
        if not (len(paragraph) > 2 and len(paragraph_text.split()) > 60):
            paragraph_text = ManuscriptEditor.convert_sentence_ends_to_newlines(
                paragraph_text
            )
            if outfile is not None:
                outfile.write(paragraph_text)
                return
            else:
                return paragraph_text, paragraph_text

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
ERROR: the paragraph below could not be revised with the AI model due to the following error:

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
        # if the first sentence of the paragraph contains the word "revised" in it,
        # then remove the entire sentence
        if paragraph.startswith("Revised:\n"):
            paragraph = paragraph.replace("Revised:\n", "", 1)
        elif paragraph.startswith("We revised the paragraph "):
            import re

            paragraph = re.sub(r"We revised the paragraph [^\n]+\n\s*", "", paragraph)

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
            return "supplementary material"
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

            current_table_paragraph = False
            last_sentence_ends_with_alphanum_or_colon = False

            for line in infile:
                if line.startswith("<!--"):
                    # This is an HTML comment.
                    while line is not None and not line.strip().endswith("-->"):
                        outfile.write(line)
                        line = next(infile, None)

                    if line is not None and line.strip().endswith("-->"):
                        outfile.write(line)
                        line = next(infile, None)

                # if line is starting either an "image paragraph", a "table
                # paragraph" or a "html comment paragraph", then skip all lines
                # until the end of that paragraph
                if line is not None and self.line_is_not_part_of_paragraph(
                    line, include_blank=False
                ):
                    if line.startswith("|"):
                        current_table_paragraph = True

                    while line is not None and line.strip() != "":
                        outfile.write(line)
                        line = next(infile, None)

                # for "table paragraphs", there is a blank line after the table
                # and then the next paragraph is the table caption that starts
                # with "Table: ". We want to include those lines as part of the
                # "table paragraph"
                if (
                    line is not None
                    and current_table_paragraph
                    and line.startswith("Table: ")
                ):
                    while line is not None and line.strip() != "":
                        outfile.write(line)
                        line = next(infile, None)

                    # we finished processing the "table paragraph"
                    current_table_paragraph = False

                # stop if we reached the end of the file
                if line is None:
                    break

                # if the line is empty and we didn't start a paragraph yet,
                # write it directly to the output file
                if line.strip() == "" and len(paragraph) == 0:
                    outfile.write(line)

                # If the line is blank, it indicates the end of a paragraph
                elif line.strip() == "":
                    # if the last sentence added to the paragraph ends with an
                    # alphanumeric character or a colon (i.e., does not end with a
                    # period as normal sentences), then we keep adding all lines to
                    # to the paragraph until we find a line with text that starts
                    # with a capital letter and is preceded by an empty line.
                    if last_sentence_ends_with_alphanum_or_colon:
                        while line is not None and not (
                            prev_line.strip() == ""
                            and (
                                (line[0].isalnum() and line[0].isupper())
                                or line.startswith("#")
                            )
                        ):
                            paragraph.append(line.strip())
                            prev_line = line
                            line = next(infile, None)

                        last_sentence_ends_with_alphanum_or_colon = False

                        # remove all trailing empty sentences in the paragraph,
                        # and for each write a new line to the output file
                        prev_line = ""
                        while len(paragraph) > 0 and paragraph[-1] == "":
                            paragraph.pop()
                            prev_line += "\n"

                    # revise and write paragraph to output file
                    self.revise_and_write_paragraph(
                        paragraph, section_name, revision_model, outfile
                    )

                    # clear the paragraph list
                    if line.strip() == "":
                        outfile.write(line)
                        paragraph = []
                    else:
                        outfile.write(prev_line)

                        if line.startswith("#"):
                            outfile.write(line)
                            paragraph = []
                        else:
                            paragraph = [line.strip()]

                # Otherwise, add the line to the paragraph list
                else:
                    line_strip = line.strip()

                    # check if line ends with a colon, a letter or a number:
                    last_sentence_ends_with_alphanum_or_colon = (
                        not self.line_is_not_part_of_paragraph(line_strip)
                        and (
                            line_strip.endswith(":")
                            or line_strip[-1].isalpha()
                            or line_strip[-1].isdigit()
                        )
                    )

                    paragraph.append(line_strip)

                prev_line = line

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

        # if specified, obtain the list of files names that have to be revised
        filenames_to_revise = None
        if env_vars.FILENAMES_TO_REVISE in os.environ:
            filenames_to_revise = os.environ[env_vars.FILENAMES_TO_REVISE]
            filenames_to_revise = {f.strip() for f in filenames_to_revise.split(",")}

        for filename in sorted(self.content_dir.glob("*.md")):
            filename_section = self.get_section_from_filename(filename.name)
            if filename_section is None:
                continue

            if (
                filenames_to_revise is not None
                and filename.name not in filenames_to_revise
            ):
                continue

            print(f"Revising file {filename.name}", flush=True)

            self.revise_file(
                filename.name,
                output_dir,
                revision_model,
                section_name=filename_section,
            )
