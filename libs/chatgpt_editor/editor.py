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
            section_name = input_filepath.stem.lower()

            if "abstract" in section_name:
                section_name = "abstract"
            elif "introduction" in section_name:
                section_name = "introduction"
            elif "results" in section_name:
                section_name = "results"
            elif "discussion" in section_name:
                section_name = "discussion"

        with open(input_filepath, "r") as infile, open(output_filepath, "w") as outfile:
            # Initialize a temporary list to store the lines of the current paragraph
            paragraph = []
            for line in infile:
                # If the line is a comment or a section name, write it directly to the output file
                if (
                    line.startswith("#")
                    or line.startswith("<!--")
                    or (line.strip() == "" and len(paragraph) == 0)
                ):
                    outfile.write(line)
                # If the line is blank, it indicates the end of a paragraph
                elif line.strip() == "":
                    # TODO: factor out the code below
                    # Process the paragraph and write it to the output file
                    paragraph_text = " ".join(paragraph)
                    paragraph_revised = revision_model.revise_paragraph(
                        paragraph_text, section_name
                    )
                    # TODO: detect whether one-line-per-sentence is used and add newlines accordingly
                    paragraph_revised = self.sentence_end_pattern.sub(
                        ".\n", paragraph_revised
                    )
                    outfile.write(paragraph_revised)
                    # Clear the paragraph list
                    paragraph = []
                # Otherwise, add the line to the paragraph list
                else:
                    paragraph.append(line)

            # If there's any remaining paragraph, process and write it to the output file
            if paragraph:
                # TODO: factor out the code below
                # Process the paragraph and write it to the output file
                paragraph_text = " ".join(paragraph)
                paragraph_revised = revision_model.revise_paragraph(
                    paragraph_text, section_name
                )
                # TODO: detect whether one-line-per-sentence is used and add newlines accordingly
                paragraph_revised = self.sentence_end_pattern.sub(
                    ".\n", paragraph_revised
                )
                outfile.write(paragraph_revised)
