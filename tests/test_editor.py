from pathlib import Path
from unittest import mock

import pytest

from manubot_ai_editor import env_vars
from manubot_ai_editor.editor import ManuscriptEditor
from manubot_ai_editor.models import (
    RandomManuscriptRevisionModel,
    DummyManuscriptRevisionModel,
    VerboseManuscriptRevisionModel,
)

MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts"


def _check_nonparagraph_lines_are_preserved(input_filepath, output_filepath):
    """
    Checks whether non-paragraph lines in the input file are preserved in the output file.
    Non-paragraph lines are those that represent section headers, comments, blank lines, as
    defined in function ManuscriptEditor.line_is_not_part_of_paragraph.
    """
    # read lines from input file
    filepath = input_filepath
    assert filepath.exists()
    with open(filepath, "r") as infile:
        input_nonpar_lines = [
            line.rstrip()
            for line in infile
            if ManuscriptEditor.line_is_not_part_of_paragraph(line)
        ]

    # read lines from output file
    filepath = output_filepath
    assert filepath.exists()
    with open(filepath, "r") as infile:
        output_nonpar_lines = [
            line.rstrip()
            for line in infile
            if ManuscriptEditor.line_is_not_part_of_paragraph(line)
        ]

    # make sure all lines that are not "paragraphs" are preserved
    for input_line in input_nonpar_lines:
        # make sure there are nonparagraph lines left in output file
        assert (
            len(output_nonpar_lines) > 0
        ), "Output file has less non-paragraph lines than input file"

        # make sure nonparagraph lines are in the same order
        assert (
            input_line == output_nonpar_lines[0]
        ), f"{input_line} != {output_nonpar_lines[0]}"

        output_nonpar_lines.remove(input_line)

    # if nonparagraph lines were preserved, then the output_nonpar_liens should be empty
    assert (
        len(output_nonpar_lines) == 0
    ), f"Non-paragraph lines are not the same: {output_nonpar_lines}"


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_abstract(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("01.abstract.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "01.abstract.md",
        output_filepath=tmp_path / "01.abstract.md",
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_introduction(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("02.introduction.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "02.introduction.md",
        output_filepath=tmp_path / "02.introduction.md",
    )


def test_get_section_from_filename():
    assert ManuscriptEditor.get_section_from_filename("00.front-matter.md") is None
    assert ManuscriptEditor.get_section_from_filename("01.abstract.md") == "abstract"
    assert (
        ManuscriptEditor.get_section_from_filename("02.introduction.md")
        == "introduction"
    )
    assert ManuscriptEditor.get_section_from_filename("02.intro.md") is None
    assert ManuscriptEditor.get_section_from_filename("03.results.md") == "results"
    assert (
        ManuscriptEditor.get_section_from_filename("04.10.results_comp.md") == "results"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("04.discussion.md") == "discussion"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("05.conclusions.md") == "conclusions"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("08.01.methods.ccc.md") == "methods"
    )
    assert (
        ManuscriptEditor.get_section_from_filename("08.15.methods.giant.md")
        == "methods"
    )
    assert ManuscriptEditor.get_section_from_filename("07.references.md") is None
    assert ManuscriptEditor.get_section_from_filename("06.acknowledgements.md") is None
    assert (
        ManuscriptEditor.get_section_from_filename("08.supplementary.md")
        == "supplementary material"
    )


@mock.patch.dict(
    "os.environ",
    {
        env_vars.SECTIONS_MAPPING: r"""
    {"02.intro.md": "introduction"}
    """
    },
)
def test_get_section_from_filename_using_environment_variable():
    assert (
        ManuscriptEditor.get_section_from_filename("02.introduction.md")
        == "introduction"
    )
    assert ManuscriptEditor.get_section_from_filename("02.intro.md") == "introduction"


@mock.patch.dict(
    "os.environ",
    {
        env_vars.SECTIONS_MAPPING: r"""
    {"02.intro.md": }
    """
    },
)
def test_get_section_from_filename_using_environment_variable_is_invalid():
    assert (
        ManuscriptEditor.get_section_from_filename("02.introduction.md")
        == "introduction"
    )
    assert ManuscriptEditor.get_section_from_filename("02.intro.md") is None


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_with_header_only(tmp_path, model):
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("04.00.results.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "04.00.results.md",
        output_filepath=tmp_path / "04.00.results.md",
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_intro_with_figure(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("04.05.results_intro.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "04.05.results_intro.md",
        output_filepath=tmp_path / "04.05.results_intro.md",
    )

    # make sure the "image paragraph" was exactly copied to the output file
    assert (
        r"""
![
**Different types of relationships in data.**
Each panel contains a set of simulated data points described by two generic variables: $x$ and $y$.
The first row shows Anscombe's quartet with four different datasets (from Anscombe I to IV) and 11 data points each.
The second row contains a set of general patterns with 100 data points each.
Each panel shows the correlation value using Pearson ($p$), Spearman ($s$) and CCC ($c$).
Vertical and horizontal red lines show how CCC clustered data points using $x$ and $y$.
](images/intro/relationships.svg "Different types of relationships in data"){#fig:datasets_rel width="100%"}
    """.strip()
        in open(tmp_path / "04.05.results_intro.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        DummyManuscriptRevisionModel(add_paragraph_marks=True),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_with_figure_without_caption(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "custom",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("00.results_image_with_no_caption.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR
        / "custom"
        / "00.results_image_with_no_caption.md",
        output_filepath=tmp_path / "00.results_image_with_no_caption.md",
    )

    # make sure the "image paragraph" was exactly copied to the output file
    assert (
        r"""
![
](images/diffs/introduction/ccc-paragraph-01.svg "Diffs - CCC introduction paragraph 01"){width="100%"}

The tool, again, significantly revised the text, producing a much better and more concise introductory paragraph.
For example, the revised first sentence (on the right) incorportes the ideas of "large datasets", and the "opportunities/possibilities" for "scientific exploration" in a clearly and briefly.
    """.strip()
        in open(tmp_path / "00.results_image_with_no_caption.md").read()
    )

    if isinstance(model, DummyManuscriptRevisionModel):
        assert (
            r"""
%%% PARAGRAPH START %%%
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC:
%%% PARAGRAPH END %%%
            """.strip()
            in open(tmp_path / "00.results_image_with_no_caption.md").read()
        )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        DummyManuscriptRevisionModel(add_paragraph_marks=True),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_results_with_table_below_nonended_paragraph(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "custom",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("00.results_table_below_nonended_paragraph.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR
        / "custom"
        / "00.results_table_below_nonended_paragraph.md",
        output_filepath=tmp_path / "00.results_table_below_nonended_paragraph.md",
    )

    # make sure the "image paragraph" was exactly copied to the output file
    assert (
        r"""
| Pathway                             | AUC   | FDR      |
|:------------------------------------|:------|:---------|
| IRIS Neutrophil-Resting             | 0.91  | 4.51e-35 |
| SVM Neutrophils                     | 0.98  | 1.43e-09 |
| PID IL8CXCR2 PATHWAY                | 0.81  | 7.04e-03 |
| SIG PIP3 SIGNALING IN B LYMPHOCYTES | 0.77  | 1.95e-02 |

Table: Pathways aligned to LV603 from the MultiPLIER models. {#tbl:sup:multiplier_pathways:lv603}

The tool, again, significantly revised the text, producing a much better and more concise introductory paragraph.
For example, the revised first sentence (on the right) incorportes the ideas of "large datasets", and the "opportunities/possibilities" for "scientific exploration" in a clearly and briefly.
    """.strip()
        in open(tmp_path / "00.results_table_below_nonended_paragraph.md").read()
    )

    if isinstance(model, DummyManuscriptRevisionModel):
        assert (
            r"""
%%% PARAGRAPH START %%%
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC.
This is the revision of the first paragraph of the introduction of CCC:
%%% PARAGRAPH END %%%
            """.strip()
            in open(tmp_path / "00.results_table_below_nonended_paragraph.md").read()
        )


def test_prepare_paragraph_with_simple_text():
    paragraph = r"""
This is the first sentence.
And this is the second one.
And this is the third and final one.
    """

    paragraph_as_list = paragraph.strip().split("\n")
    paragraph_as_list = [sentence.strip() for sentence in paragraph_as_list]
    assert len(paragraph_as_list) == 3

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    prepared_paragraph = me.prepare_paragraph(paragraph_as_list)
    assert " ".join(paragraph_as_list) + "\n" == prepared_paragraph


def test_prepare_paragraph_with_introduction_text():
    paragraph = r"""
In transcriptomics, many analyses start with estimating the correlation between genes.
More sophisticated approaches built on correlation analysis can suggest gene function [@pmid:21241896], aid in discovering common and cell lineage-specific regulatory networks [@pmid:25915600], and capture important interactions in a living organism that can uncover molecular mechanisms in other species [@pmid:21606319; @pmid:16968540].
The analysis of large RNA-seq datasets [@pmid:32913098; @pmid:34844637] can also reveal complex transcriptional mechanisms underlying human diseases [@pmid:27479844; @pmid:31121115; @pmid:30668570; @pmid:32424349; @pmid:34475573].
    """

    paragraph_as_list = paragraph.strip().split("\n")
    paragraph_as_list = [sentence.strip() for sentence in paragraph_as_list]
    assert len(paragraph_as_list) == 3

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    prepared_paragraph = me.prepare_paragraph(paragraph_as_list)
    assert " ".join(paragraph_as_list) + "\n" == prepared_paragraph


def test_prepare_paragraph_with_equations():
    # from PhenoPLIER manuscript:
    paragraph_as_list = r"""
S-PrediXcan [@doi:10.1038/s41467-018-03621-1] is the summary version of PrediXcan [@doi:10.1038/ng.3367].
PrediXcan models the trait as a linear function of the gene's expression on a single tissue using the univariate model

$$
\mathbf{y} = \mathbf{t}_l \gamma_l + \bm{\epsilon}_l,
$$ {#eq:predixcan}

where $\hat{\gamma}_l$ is the estimated effect size or regression coefficient, and $\bm{\epsilon}_l$ are the error terms with variance $\sigma_{\epsilon}^{2}$.
The significance of the association is assessed by computing the $z$-score $\hat{z}_{l}=\hat{\gamma}_l / \mathrm{se}(\hat{\gamma}_l)$ for a gene's tissue model $l$.
PrediXcan needs individual-level data to fit this model, whereas S-PrediXcan approximates PrediXcan $z$-scores using only GWAS summary statistics with the expression

$$
\hat{z}_{l} \approx \sum_{a \in model_{l}} w_a^l \frac{\hat{\sigma}_a}{\hat{\sigma}_l} \frac{\hat{\beta}_a}{\mathrm{se}(\hat{\beta}_a)},
$$ {#eq:spredixcan}

where $\hat{\sigma}_a$ is the variance of SNP $a$, $\hat{\sigma}_l$ is the variance of the predicted expression of a gene in tissue $l$, and $\hat{\beta}_a$ is the estimated effect size of SNP $a$ from the GWAS.
In these TWAS methods, the genotype variances and covariances are always estimated using the Genotype-Tissue Expression project (GTEx v8) [@doi:10.1126/science.aaz1776] as the reference panel.
Since S-PrediXcan provides tissue-specific direction of effects (for instance, whether a higher or lower predicted expression of a gene confers more or less disease risk), we used the $z$-scores in our drug repurposing approach (described below).
    """.strip().split(
        "\n"
    )
    paragraph_as_list = [sentence.strip() for sentence in paragraph_as_list]
    assert len(paragraph_as_list) == 18

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    prepared_paragraph = me.prepare_paragraph(paragraph_as_list)
    assert (
        r"""S-PrediXcan [@doi:10.1038/s41467-018-03621-1] is the summary version of PrediXcan [@doi:10.1038/ng.3367]. PrediXcan models the trait as a linear function of the gene's expression on a single tissue using the univariate model

$$
\mathbf{y} = \mathbf{t}_l \gamma_l + \bm{\epsilon}_l,
$$ {#eq:predixcan}

where $\hat{\gamma}_l$ is the estimated effect size or regression coefficient, and $\bm{\epsilon}_l$ are the error terms with variance $\sigma_{\epsilon}^{2}$. The significance of the association is assessed by computing the $z$-score $\hat{z}_{l}=\hat{\gamma}_l / \mathrm{se}(\hat{\gamma}_l)$ for a gene's tissue model $l$. PrediXcan needs individual-level data to fit this model, whereas S-PrediXcan approximates PrediXcan $z$-scores using only GWAS summary statistics with the expression

$$
\hat{z}_{l} \approx \sum_{a \in model_{l}} w_a^l \frac{\hat{\sigma}_a}{\hat{\sigma}_l} \frac{\hat{\beta}_a}{\mathrm{se}(\hat{\beta}_a)},
$$ {#eq:spredixcan}

where $\hat{\sigma}_a$ is the variance of SNP $a$, $\hat{\sigma}_l$ is the variance of the predicted expression of a gene in tissue $l$, and $\hat{\beta}_a$ is the estimated effect size of SNP $a$ from the GWAS. In these TWAS methods, the genotype variances and covariances are always estimated using the Genotype-Tissue Expression project (GTEx v8) [@doi:10.1126/science.aaz1776] as the reference panel. Since S-PrediXcan provides tissue-specific direction of effects (for instance, whether a higher or lower predicted expression of a gene confers more or less disease risk), we used the $z$-scores in our drug repurposing approach (described below).
"""
        == prepared_paragraph
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        DummyManuscriptRevisionModel(add_paragraph_marks=True),
    ],
)
def test_revise_methods_paragraph_with_too_few_sentences_or_words(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    me.revise_file("07.00.methods.md", tmp_path, model)

    # make sure paragraph was not processed
    # first, that original text is there (makes sense for the RandomManuscriptRevisionModel)
    assert (
        r"""
Since the gold standard of drug-disease medical indications is described with Disease Ontology IDs (DOID) [@doi:10.1093/nar/gky1032], we mapped PhenomeXcan traits to the Experimental Factor Ontology [@doi:10.1093/bioinformatics/btq099] using [@url:https://github.com/EBISPOT/EFO-UKB-mappings], and then to DOID.
"""
        in open(tmp_path / "07.00.methods.md").read()
    )

    # second, that it was not processed as part of a paragraph (makes sense for the DummyManuscriptRevisionModel)
    assert (
        r"""
%%% PARAGRAPH START %%%
Since the gold standard of drug-disease medical indications is described with Disease Ontology IDs (DOID) [@doi:10.1093/nar/gky1032], we mapped PhenomeXcan traits to the Experimental Factor Ontology [@doi:10.1093/bioinformatics/btq099] using [@url:https://github.com/EBISPOT/EFO-UKB-mappings], and then to DOID.
%%% PARAGRAPH END %%%
"""
        not in open(tmp_path / "07.00.methods.md").read()
    )

    # and same for another paragraph
    assert (
        r"""
We ran our regression model for all 987 LVs across the 4,091 traits in PhenomeXcan.
For replication, we ran the model in the 309 phecodes in eMERGE.
We adjusted the $p$-values using the Benjamini-Hochberg procedure.
"""
        in open(tmp_path / "07.00.methods.md").read()
    )

    assert (
        r"""
%%% PARAGRAPH START %%%
We ran our regression model for all 987 LVs across the 4,091 traits in PhenomeXcan.
For replication, we ran the model in the 309 phecodes in eMERGE.
We adjusted the $p$-values using the Benjamini-Hochberg procedure.
%%% PARAGRAPH END %%%
"""
        not in open(tmp_path / "07.00.methods.md").read()
    )


@pytest.mark.parametrize(
    "model,filename",
    [
        (DummyManuscriptRevisionModel(add_paragraph_marks=True), "07.00.methods.md"),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_methods_with_equation(tmp_path, model, filename):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file(filename, tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "phenoplier" / filename,
        output_filepath=tmp_path / filename,
    )

    # make sure regular paragraphs are correctly marked
    assert (
        r"""
%%% PARAGRAPH START %%%
PhenoPLIER is a framework that combines different computational approaches to integrate gene-trait associations and drug-induced transcriptional responses with groups of functionally-related genes (referred to as gene modules or latent variables/LVs).
Gene-trait associations are computed using the PrediXcan family of methods, whereas latent variables are inferred by the MultiPLIER models applied on large gene expression compendia.
PhenoPLIER provides 1) a regression model to compute an LV-trait association, 2) a consensus clustering approach applied to the latent space to learn shared and distinct transcriptomic properties between traits, and 3) an interpretable, LV-based drug repurposing framework.
We provide the details of these methods below.
%%% PARAGRAPH END %%%
        """.strip()
        in open(tmp_path / filename).read()
    )

    # make sure the "equation paragraph" is correctly marked
    assert (
        r"""
%%% PARAGRAPH START %%%
S-PrediXcan [@doi:10.1038/s41467-018-03621-1] is the summary version of PrediXcan [@doi:10.1038/ng.3367].
PrediXcan models the trait as a linear function of the gene's expression on a single tissue using the univariate model

$$
\mathbf{y} = \mathbf{t}_l \gamma_l + \bm{\epsilon}_l,
$$ {#eq:predixcan}

where $\hat{\gamma}_l$ is the estimated effect size or regression coefficient, and $\bm{\epsilon}_l$ are the error terms with variance $\sigma_{\epsilon}^{2}$.
The significance of the association is assessed by computing the $z$-score $\hat{z}_{l}=\hat{\gamma}_l / \mathrm{se}(\hat{\gamma}_l)$ for a gene's tissue model $l$.
PrediXcan needs individual-level data to fit this model, whereas S-PrediXcan approximates PrediXcan $z$-scores using only GWAS summary statistics with the expression

$$
\hat{z}_{l} \approx \sum_{a \in model_{l}} w_a^l \frac{\hat{\sigma}_a}{\hat{\sigma}_l} \frac{\hat{\beta}_a}{\mathrm{se}(\hat{\beta}_a)},
$$ {#eq:spredixcan}

where $\hat{\sigma}_a$ is the variance of SNP $a$, $\hat{\sigma}_l$ is the variance of the predicted expression of a gene in tissue $l$, and $\hat{\beta}_a$ is the estimated effect size of SNP $a$ from the GWAS.
In these TWAS methods, the genotype variances and covariances are always estimated using the Genotype-Tissue Expression project (GTEx v8) [@doi:10.1126/science.aaz1776] as the reference panel.
Since S-PrediXcan provides tissue-specific direction of effects (for instance, whether a higher or lower predicted expression of a gene confers more or less disease risk), we used the $z$-scores in our drug repurposing approach (described below).
%%% PARAGRAPH END %%%
    """.strip()
        in open(tmp_path / filename).read()
    )


@pytest.mark.parametrize(
    "model,filename",
    [
        (
            DummyManuscriptRevisionModel(add_paragraph_marks=True),
            "07.00.methods_already_revised.md",
        ),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_methods_with_equation_that_was_alrady_revised(
    tmp_path, model, filename
):
    """
    This test was added after getting an error in processing a file that was already
    processed by the manuscript editor. The error was in ManuscriptEditor.prepare_paragraph
    when a None sentence was added to the equation_sentences list.
    """
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file(filename, tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "phenoplier" / filename,
        output_filepath=tmp_path / filename,
    )


@pytest.mark.parametrize(
    "model,filename",
    [
        (
            DummyManuscriptRevisionModel(add_paragraph_marks=True),
            "05.methods.md",
        ),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_methods_mutator_epistasis_paper(
    tmp_path, model, filename
):
    """
    This papers has several test cases:
     - it ends with multiple blank lines
     - it uses block of source code or bullet points, sometimes preceded by a paragraph with a colon (:)
    """
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "mutator-epistasis",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file(filename, tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "mutator-epistasis" / filename,
        output_filepath=tmp_path / filename,
    )

    assert (
            r"""
%%% PARAGRAPH START %%%
Briefly, we identified private single-nucleotide mutations in each BXD that were absent from all other BXDs, as well as from the C57BL/6J and DBA/2J parents.
We required each private variant to be meet the following criteria:

* genotyped as either homozygous or heterozygous for the alternate allele, with at least 90% of sequencing reads supporting the alternate allele

* supported by at least 10 sequencing reads

* Phred-scaled genotype quality of at least 20

* must not overlap regions of the genome annotated as segmental duplications or simple repeats in GRCm38/mm10

* must occur on a parental haplotype that was inherited by at least one other BXD at the same locus; these other BXDs must be homozygous for the reference allele at the variant site
%%% PARAGRAPH END %%%
        """.strip()
            in open(tmp_path / filename).read()
    )
    
    assert (
            r"""
### Extracting mutation signatures 

We used SigProfilerExtractor (v.1.1.21) [@PMID:30371878] to extract mutation signatures from the BXD mutation data.
After converting the BXD mutation data to the "matrix" input format expected by SigProfilerExtractor, we ran the `sigProfilerExtractor` method as follows:

```python
# install the mm10 mouse reference data
genInstall.install('mm10')

# run mutation signature extraction
sig.sigProfilerExtractor(
    'matrix',
    /path/to/output/directory,
    /path/to/input/mutations,
    maximum_signatures=10,
    nmf_replicates=100,
    opportunity_genome="mm10",
)
```

### Comparing mutation spectra between Mouse Genomes Project strains
        """.strip()
            in open(tmp_path / filename).read()
    )
    
    assert (
            r"""
%%% PARAGRAPH START %%%
We investigated the region implicated by our aggregate mutation spectrum distance approach on chromosome 6 by subsetting the joint-genotyped BXD VCF file (European Nucleotide Archive accession PRJEB45429 [@url:https://www.ebi.ac.uk/ena/browser/view/PRJEB45429]) using `bcftools` [@PMID:33590861].
We defined the candidate interval surrounding the cosine distance peak on chromosome 6 as the 90% bootstrap confidence interval (extending from approximately 95 Mbp to 114 Mbp).
To predict the functional impacts of both single-nucleotide variants and indels on splicing, protein structure, etc., we annotated variants in the BXD VCF using the following `snpEff` [@PMID:22728672] command:
%%% PARAGRAPH END %%%

```
 java -Xmx16g -jar /path/to/snpeff/jarfile GRCm38.75 /path/to/bxd/vcf > /path/to/uncompressed/output/vcf
```
        """.strip()
            in open(tmp_path / filename).read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_supplementary_material_with_tables_and_multiline_html_comments(
    tmp_path, model
):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("20.00.supplementary_material.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR / "ccc" / "20.00.supplementary_material.md",
        output_filepath=tmp_path / "20.00.supplementary_material.md",
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
| | **Interaction confidence** <!-- $colspan="7" -->    | | | | | | |
|:------:|:-----:|:-----:|:-----:|:--------:|:-----:|:-----:|:-----:|
| | **Blood** <!-- $colspan="3" --> | | | **Predicted cell type** <!-- $colspan="4" --> | | | |
| **Gene** |  **Min.** | **Avg.** | **Max.** |  **Cell type** | **Min.** | **Avg.** | **Max.** |
| *IFNG* | 0.19 | 0.42 | 0.54 | Natural killer cell<!-- $rowspan="2" --> | 0.74 | 0.90 | 0.99 |
| *SDS* | 0.18 | 0.29 | 0.41 | 0.65 | 0.81 | 0.94<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *JUN* | 0.26 | 0.68 | 0.97 | Mononuclear phagocyte<!-- $rowspan="2" --> | 0.36 | 0.73 | 0.94 |
| *APOC1* | 0.22 | 0.47 | 0.77 | 0.29 | 0.50 | 0.80<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *ZDHHC12* | 0.05 | 0.07 | 0.10 | Macrophage<!-- $rowspan="2" --> | 0.03 | 0.12 | 0.33 |
| *CCL18* | 0.74 | 0.79 | 0.86 | 0.36 | 0.70 | 0.90<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *RASSF2* | 0.69 | 0.77 | 0.90 | Leukocyte<!-- $rowspan="2" --> | 0.66 | 0.74 | 0.88 |
| *CYTIP* | 0.74 | 0.85 | 0.91 | 0.76 | 0.84 | 0.91<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *MYOZ1* | 0.09 | 0.17 | 0.37 | Skeletal muscle<!-- $rowspan="2" --> | 0.11 | 0.11 | 0.12 |
| *TNNI2* | 0.10 | 0.22 | 0.44 | 0.10 | 0.11 | 0.12<!-- $removenext="2" --> |
| <!-- $colspan="7" --> |||||||
| *PYGM* | 0.02 | 0.04 | 0.14 | Skeletal muscle<!-- $rowspan="2" --> | 0.01 | 0.02 | 0.04 |
| *TPM2* | 0.05 | 0.56 | 0.80 | 0.01 | 0.28 | 0.47<!-- $removenext="2" --> |

Table: Network statistics of six gene pairs shown in Figure @fig:upsetplot_coefs b for blood and predicted cell types.
Only gene pairs present in GIANT models are listed.
For each gene in the pair (first column), the minimum, average and maximum interaction coefficients with the other genes in the network are shown.
{#tbl:giant:weights}
    """.strip()
        in open(tmp_path / "20.00.supplementary_material.md").read()
    )

    # make sure the "HTML comment paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- ![
**Predicted tissue-specific networks from GIANT for six gene pairs prioritized by correlation coefficients.**
Gene pairs are from Figure @fig:upsetplot_coefs b.
A node represents a gene and an edge the probability that two genes are part of the same biological process in a specific cell type.
The cell type for each gene network was automatically predicted using [@doi:10.1101/gr.155697.113], and it is indicated at the top-right corner of each network.
A maximum of 15 genes are shown for each subfigure.
The GIANT web application automatically determined a minimum interaction confidence (edges' weights) to be shown.
All these analyses can be performed online using the following links:
*IFNG* - *SDS* [@url:https://hb.flatironinstitute.org/gene/10993+3458],
*JUN* - *APOC1* [@url:https://hb.flatironinstitute.org/gene/3725+341],
*ZDHHC12* - *CCL18* [@url:https://hb.flatironinstitute.org/gene/6362+84885],
*RASSF2* - *CYTIP* [@url:https://hb.flatironinstitute.org/gene/9770+9595],
*MYOZ1* - *TNNI2* [@url:https://hb.flatironinstitute.org/gene/58529+7136],
*PYGM* - *TPM2* [@url:https://hb.flatironinstitute.org/gene/5837+7169].
The GIANT web-server was accessed on April 4, 2022.
](images/coefs_comp/giant_networks/auto_selected_tissues/main.svg "GIANT network interaction"){#fig:giant_gene_pairs:pred_tissue width="100%"} -->
    """.strip()
        in open(tmp_path / "20.00.supplementary_material.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_supplementary_material_from_phenoplier_with_many_tables(
    tmp_path, model
):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("50.00.supplementary_material.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR
        / "phenoplier"
        / "50.00.supplementary_material.md",
        output_filepath=tmp_path / "50.00.supplementary_material.md",
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:multiplier_pathways:start -->
| Pathway                             | AUC   | FDR      |
|:------------------------------------|:------|:---------|
| IRIS Neutrophil-Resting             | 0.91  | 4.51e-35 |
| SVM Neutrophils                     | 0.98  | 1.43e-09 |
| PID IL8CXCR2 PATHWAY                | 0.81  | 7.04e-03 |
| SIG PIP3 SIGNALING IN B LYMPHOCYTES | 0.77  | 1.95e-02 |

Table: Pathways aligned to LV603 from the MultiPLIER models. {#tbl:sup:multiplier_pathways:lv603}
<!-- LV603:multiplier_pathways:end -->
    """.strip()
        in open(tmp_path / "50.00.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:phenomexcan_traits_assocs:start -->
| Trait description                         | Sample size   | Cases   | FDR            |
|:------------------------------------------|:--------------|:--------|:---------------|
| Basophill percentage                      | 349,861       |         | 1.19e&#8209;10 |
| Basophill count                           | 349,856       |         | 1.89e&#8209;05 |
| Treatment/medication code: ispaghula husk | 361,141       | 327     | 1.36e&#8209;02 |

Table: Significant trait associations of LV603 in PhenomeXcan. {#tbl:sup:phenomexcan_assocs:lv603}
<!-- LV603:phenomexcan_traits_assocs:end -->
    """.strip()
        in open(tmp_path / "50.00.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:emerge_traits_assocs:start -->
| Phecode                     | Trait description   | Sample size   | Cases   | FDR   |
|:----------------------------|:--------------------|:--------------|:--------|:------|
| No significant associations |                     |               |         |       |

Table: Significant trait associations of LV603 in eMERGE. {#tbl:sup:emerge_assocs:lv603}
<!-- LV603:emerge_traits_assocs:end -->
    """.strip()
        in open(tmp_path / "50.00.supplementary_material.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_supplementary_material_from_phenoplier_with_many_tables_and_complex_html_comments(
    tmp_path, model
):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "phenoplier",
    )

    model.title = me.title
    model.keywords = me.keywords

    me.revise_file("50.01.supplementary_material.md", tmp_path, model)

    _check_nonparagraph_lines_are_preserved(
        input_filepath=MANUSCRIPTS_DIR
        / "phenoplier"
        / "50.01.supplementary_material.md",
        output_filepath=tmp_path / "50.01.supplementary_material.md",
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:multiplier_pathways:start
this is a more complex multiline html comment -->
| Pathway                             | AUC   | FDR      |
|:------------------------------------|:------|:---------|
| IRIS Neutrophil-Resting             | 0.91  | 4.51e-35 |
| SVM Neutrophils                     | 0.98  | 1.43e-09 |
| PID IL8CXCR2 PATHWAY                | 0.81  | 7.04e-03 |
| SIG PIP3 SIGNALING IN B LYMPHOCYTES | 0.77  | 1.95e-02 |

Table: Pathways aligned to LV603 from the MultiPLIER models. {#tbl:sup:multiplier_pathways:lv603}
<!-- LV603:multiplier_pathways:end -->
    """.strip()
        in open(tmp_path / "50.01.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!-- LV603:phenomexcan_traits_assocs:start
and this html comments is multiline but

also has an empty line in the middle-->
| Trait description                         | Sample size   | Cases   | FDR            |
|:------------------------------------------|:--------------|:--------|:---------------|
| Basophill percentage                      | 349,861       |         | 1.19e&#8209;10 |
| Basophill count                           | 349,856       |         | 1.89e&#8209;05 |
| Treatment/medication code: ispaghula husk | 361,141       | 327     | 1.36e&#8209;02 |

Table: Significant trait associations of LV603 in PhenomeXcan. {#tbl:sup:phenomexcan_assocs:lv603}
<!-- LV603:phenomexcan_traits_assocs:end 


-->   
    """.strip()
        in open(tmp_path / "50.01.supplementary_material.md").read()
    )

    # make sure the "table paragraph" was exactly copied to the output file
    assert (
        r"""
<!--

and this html multiline comment has a space
LV603:emerge_traits_assocs:start

-->
| Phecode                     | Trait description   | Sample size   | Cases   | FDR   |
|:----------------------------|:--------------------|:--------------|:--------|:------|
| No significant associations |                     |               |         |       |

Table: Significant trait associations of LV603 in eMERGE. {#tbl:sup:emerge_assocs:lv603}
<!-- LV603:emerge_traits_assocs:end -->
    """.strip()
        in open(tmp_path / "50.01.supplementary_material.md").read()
    )


@pytest.mark.parametrize(
    "model",
    [
        VerboseManuscriptRevisionModel("Revised:\n"),
        VerboseManuscriptRevisionModel("Revised:\n\n"),
        VerboseManuscriptRevisionModel(
            "We revised the paragraph from the Methods section of the academic paper titled 'Projecting genetic associations through gene expression patterns highlights disease etiology and drug mechanisms' as follows:\n\n"
        ),
    ],
)
def test_revise_section_where_model_says_what_it_is_doing(model):
    # sometimes, the GPT model returns an initial paragraph saying "We revised"
    # or "Revised:". Here I make sure the editor does not include this paragraph.
    orig_paragraph_text = r"""
This is a paragraph that was not revised.
And this is the second line of the same paragraph.
And finally, a third sentence so we have more than 2.
And finally, a third sentence so we have more than 2.
And finally, a third sentence so we have more than 2.
And finally, a third sentence so we have more than 2.
        """

    # make sure I have the minimum number of words to pass the "too short" check
    assert len(orig_paragraph_text.split()) > 60, len(orig_paragraph_text.split())

    paragraph = orig_paragraph_text.strip().split("\n")
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 6

    model.title = "Projecting genetic associations through gene expression patterns highlights disease etiology and drug mechanisms"
    model.keywords = [
        "genetic studies",
        "functional genomics",
        "gene co-expression",
        "therapeutic targets",
        "drug repurposing",
        "clustering of complex traits",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph,
        model,
        "methods",
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)

    # the paragraph does not contain the "Revised:" or "We revised" initial
    assert orig_paragraph_text.strip() == paragraph_revised.strip()


@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 12


@mock.patch.dict(
    "os.environ",
    {
        env_vars.FILENAMES_TO_REVISE: r"01.abstract.md,02.introduction.md",
    },
)
@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript_only_some_files_are_selected(tmp_path, model):
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 2
    output_md_filenames = [f.name for f in output_md_files]
    assert "01.abstract.md" in output_md_filenames
    assert "02.introduction.md" in output_md_filenames


@mock.patch.dict(
    "os.environ",
    {
        env_vars.FILENAMES_TO_REVISE: "",
    },
)
@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript_list_of_selected_files_is_empty(tmp_path, model):
    # in this case, the list of selected files is ignored and all files are
    # revised
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 12


@mock.patch.dict(
    "os.environ",
    {
        env_vars.FILENAMES_TO_REVISE: "",
    },
)
@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript_non_standard_filenames_without_custom_prompt(
    tmp_path, model
):
    # in this case, the list of selected files is empty and files have non standard names,
    # so none of them are revised
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc_non_standard_filenames",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 0


@mock.patch.dict(
    "os.environ",
    {
        env_vars.FILENAMES_TO_REVISE: "",
        env_vars.CUSTOM_PROMPT: "proofread and revise the following paragraph with manuscript title '{title}': {paragraph_text}",
    },
)
@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript_non_standard_filenames_with_custom_prompt(
    tmp_path, model
):
    # in this case, the list of selected files is empty but there is a custom prompt, so all files are revised
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc_non_standard_filenames",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 6


@mock.patch.dict(
    "os.environ",
    {env_vars.FILENAMES_TO_REVISE: "", env_vars.CUSTOM_PROMPT: ""},
)
@pytest.mark.parametrize(
    "model",
    [
        RandomManuscriptRevisionModel(),
        # GPT3CompletionModel(None, None),
    ],
)
def test_revise_entire_manuscript_non_standard_filenames_with_empty_custom_prompt(
    tmp_path, model
):
    # in this case, the list of selected files is empty and the custom prompt env variable is there but it's empty,
    # this use case is when the custom prompt is not provided in the workflow interface
    print(f"\n{str(tmp_path)}\n")

    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc_non_standard_filenames",
    )

    model.title = me.title
    model.keywords = me.keywords

    output_folder = tmp_path
    assert output_folder.exists()

    me.revise_manuscript(output_folder, model)

    output_md_files = list(output_folder.glob("*.md"))
    assert len(output_md_files) == 0
