import re
from pathlib import Path

from chatgpt_editor.models import ManuscriptRevisionModel
from chatgpt_editor.utils import get_yaml_field


class ManuscriptEditor:
    def __init__(self, content_dir: str | Path):
        self.content_dir = Path(content_dir)

        metadata_file = self.content_dir / "metadata.yaml"
        assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"
        self.title = get_yaml_field(metadata_file, "title")
        self.keywords = get_yaml_field(metadata_file, "keywords")

        self.sentence_end_pattern = re.compile(r"\. ")

    @staticmethod
    def line_is_not_part_of_paragraph(line: str) -> bool:
        return line.startswith("#") or line.startswith("<!--") or line.strip() == ""

    def revise_and_write_paragraph(
        self,
        paragraph: list[str],
        section_name: str,
        revision_model: ManuscriptRevisionModel,
        outfile,
    ):
        """
        Revises and writes a paragraph to the output file.

        Arguments:
            paragraph: list of lines of the paragraph.
            section_name: name of the section the paragraph belongs to.
            revision_model: model to use for revision.
            outfile: file object to write the revised paragraph to.
        """
        # Process the paragraph and revise it with model
        paragraph_text = "".join(paragraph)
        paragraph_revised = revision_model.revise_paragraph(
            paragraph_text, section_name
        )

        # put sentences into new lines
        paragraph_revised = self.sentence_end_pattern.sub(
            ".\n", paragraph_revised
        )

        outfile.write(paragraph_revised + "\n")

    def get_section_from_filename(self, filename: str) -> str:
        """
        Returns the section name of a file based on its filename.
        """
        filename = filename.lower()

        if "abstract" in filename:
            return "abstract"
        elif "introduction" in filename:
            return "introduction"
        elif "results" in filename:
            return "results"
        elif "discussion" in filename:
            return "discussion"
        else:
            return None

    def revise_file(
        self,
        input_filename: str,
        output_dir: Path | str,
        revision_model: ManuscriptRevisionModel,
        section_name: str = None,
    ):
        """
        TODO: add docstring

        Args:
            input_filename (str):
            output_dir (Path | str):
            revision_model (ManuscriptRevisionModel):
            section_name (str, optional): Defaults to None.
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
                # if the line is a comment or a section name, write it directly
                # to the output file
                if self.line_is_not_part_of_paragraph(line) and len(paragraph) == 0:
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
                    paragraph.append(line)

            # If there's any remaining paragraph, process and write it to the
            # output file
            if paragraph:
                self.revise_and_write_paragraph(
                    paragraph, section_name, revision_model, outfile
                )
