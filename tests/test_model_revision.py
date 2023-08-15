"""
These tests need to call the OpenAI API, so they are in a separate file and can incur costs.
"""
import difflib
from unittest import mock

import pytest

from manubot_ai_editor import env_vars
from manubot_ai_editor.editor import ManuscriptEditor
from manubot_ai_editor.models import GPT3CompletionModel
from manubot_ai_editor.utils import starts_with_similar


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_abstract_ccc(model):
    # from CCC manuscript
    paragraph = r"""
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
Here we introduce the Clustermatch Correlation Coefficient (CCC), an efficient, easy-to-use and not-only-linear coefficient based on machine learning models.
CCC reveals biologically meaningful linear and nonlinear patterns missed by standard, linear-only correlation coefficients.
CCC captures general patterns in data by comparing clustering solutions while being much faster than state-of-the-art coefficients such as the Maximal Information Coefficient.
When applied to human gene expression data, CCC identifies robust linear relationships while detecting nonlinear patterns associated, for example, with sex differences that are not captured by linear-only coefficients.
Gene pairs highly ranked by CCC were enriched for interactions in integrated networks built from protein-protein interaction, transcription factor regulation, and chemical and genetic perturbations, suggesting that CCC could detect functional relationships that linear-only methods missed.
CCC is a highly-efficient, next-generation not-only-linear correlation coefficient that can readily be applied to genome-scale data and other domains across different data types.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 8

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "abstract"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # # original and revised paragraph should be quite different
    # _ratio = difflib.SequenceMatcher(lambda x: x in (" ", "\n",), paragraph_text, paragraph_revised).ratio()
    # assert _ratio < 0.10 if model.endpoint != "edits" else 1.0, _ratio

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[" not in paragraph_revised
    assert "@" not in paragraph_revised

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@mock.patch.dict(
    "os.environ",
    {env_vars.CUSTOM_PROMPT: "proofread the following paragraph"},
)
@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_abstract_ccc_with_custom_prompt(model):
    # from CCC manuscript
    paragraph = r"""
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
Here we introduce the Clustermatch Correlation Coefficient (CCC), an efficient, easy-to-use and not-only-linear coefficient based on machine learning models.
CCC reveals biologically meaningful linear and nonlinear patterns missed by standard, linear-only correlation coefficients.
CCC captures general patterns in data by comparing clustering solutions while being much faster than state-of-the-art coefficients such as the Maximal Information Coefficient.
When applied to human gene expression data, CCC identifies robust linear relationships while detecting nonlinear patterns associated, for example, with sex differences that are not captured by linear-only coefficients.
Gene pairs highly ranked by CCC were enriched for interactions in integrated networks built from protein-protein interaction, transcription factor regulation, and chemical and genetic perturbations, suggesting that CCC could detect functional relationships that linear-only methods missed.
CCC is a highly-efficient, next-generation not-only-linear correlation coefficient that can readily be applied to genome-scale data and other domains across different data types.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 8

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "abstract"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # # since the custom prompt also "proofreads", the similarity between input and revised text should be very high
    # _ratio = difflib.SequenceMatcher(lambda x: x in (" ", "\n",), paragraph_text, paragraph_revised).ratio()
    # assert _ratio > 0.50, _ratio

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[" not in paragraph_revised
    assert "@" not in paragraph_revised

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_abstract_phenoplier(model):
    # from PhenoPLIER manuscript
    paragraph = r"""
Genes act in concert with each other in specific contexts to perform their functions.
Determining how these genes influence complex traits requires a mechanistic understanding of expression regulation across different conditions.
It has been shown that this insight is critical for developing new therapies.
In this regard, the role of individual genes in disease-relevant mechanisms can be hypothesized with transcriptome-wide association studies (TWAS), which have represented a significant step forward in testing the mediating role of gene expression in GWAS associations.
However, modern models of the architecture of complex traits predict that gene-gene interactions play a crucial role in disease origin and progression.
Here we introduce PhenoPLIER, a computational approach that maps gene-trait associations and pharmacological perturbation data into a common latent representation for a joint analysis.
This representation is based on modules of genes with similar expression patterns across the same conditions.
We observed that diseases were significantly associated with gene modules expressed in relevant cell types, and our approach was accurate in predicting known drug-disease pairs and inferring mechanisms of action.
Furthermore, using a CRISPR screen to analyze lipid regulation, we found that functionally important players lacked TWAS associations but were prioritized in trait-associated modules by PhenoPLIER.
By incorporating groups of co-expressed genes, PhenoPLIER can contextualize genetic associations and reveal potential targets missed by single-gene strategies.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 10

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
        paragraph, model, "abstract"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[" not in paragraph_revised
    assert "@" not in paragraph_revised

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_abstract_ai_revision(model):
    # from LLM for articles revision manuscript
    paragraph = r"""
Academics often communicate through scholarly manuscripts.
These manuscripts describe new advances, summarize existing literature, or argue for changes in the status quo.
Writing and revising manuscripts can be a time-consuming process.
Large language models are bringing new capabilities to many areas of knowledge work.
We integrated the use of large language models into the Manubot publishing ecosystem.
Users of Manubot can run a workflow, which will trigger a series of queries to OpenAI's language models, produce revisions, and create a timestamped set of suggested revisions.
Given the amount of time that researchers put into crafting prose, we expect this advance to radically transform the type of knowledge work that academics perform.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 7

    model.title = "A publishing infrastructure for AI-assisted academic authoring"
    model.keywords = [
        "manubot",
        "artificial intelligence",
        "scholarly publishing",
        "software",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "abstract"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[" not in paragraph_revised
    assert "@" not in paragraph_revised

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_introduction_paragraph_with_single_and_multiple_citations_together(
    model,
):
    # from CCC manuscript
    paragraph = r"""
In transcriptomics, many analyses start with estimating the correlation between genes.
More sophisticated approaches built on correlation analysis can suggest gene function [@pmid:21241896], aid in discovering common and cell lineage-specific regulatory networks [@pmid:25915600], and capture important interactions in a living organism that can uncover molecular mechanisms in other species [@pmid:21606319; @pmid:16968540].
The analysis of large RNA-seq datasets [@pmid:32913098; @pmid:34844637] can also reveal complex transcriptional mechanisms underlying human diseases [@pmid:27479844; @pmid:31121115; @pmid:30668570; @pmid:32424349; @pmid:34475573].
Since the introduction of the omnigenic model of complex traits [@pmid:28622505; @pmid:31051098], gene-gene relationships are playing an increasingly important role in genetic studies of human diseases [@pmid:34845454; @doi:10.1101/2021.07.05.450786; @doi:10.1101/2021.10.21.21265342; @doi:10.1038/s41588-021-00913-z], even in specific fields such as polygenic risk scores [@doi:10.1016/j.ajhg.2021.07.003].
In this context, recent approaches combine disease-associated genes from genome-wide association studies (GWAS) with gene co-expression networks to prioritize "core" genes directly affecting diseases [@doi:10.1186/s13040-020-00216-9; @doi:10.1101/2021.07.05.450786; @doi:10.1101/2021.10.21.21265342].
These core genes are not captured by standard statistical methods but are believed to be part of highly-interconnected, disease-relevant regulatory networks.
Therefore, advanced correlation coefficients could immediately find wide applications across many areas of biology, including the prioritization of candidate drug targets in the precision medicine field.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 7

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "introduction"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[@" in paragraph_revised
    assert paragraph_revised.count("@") >= int(paragraph_text.count("@") * 0.50)

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_introduction_paragraph_with_citations_and_paragraph_is_the_first(model):
    # from PhenoPLIER manuscript
    paragraph = r"""
Genes work together in context-specific networks to carry out different functions [@pmid:19104045; @doi:10.1038/ng.3259].
Variations in these genes can change their functional role and, at a higher level, affect disease-relevant biological processes [@doi:10.1038/s41467-018-06022-6].
In this context, determining how genes influence complex traits requires mechanistically understanding expression regulation across different cell types [@doi:10.1126/science.aaz1776; @doi:10.1038/s41586-020-2559-3; @doi:10.1038/s41576-019-0200-9], which in turn should lead to improved treatments [@doi:10.1038/ng.3314; @doi:10.1371/journal.pgen.1008489].
Previous studies have described different regulatory DNA elements [@doi:10.1038/nature11247; @doi:10.1038/nature14248; @doi:10.1038/nature12787; @doi:10.1038/s41586-020-03145-z; @doi:10.1038/s41586-020-2559-3] including genetic effects on gene expression across different tissues [@doi:10.1126/science.aaz1776].
Integrating functional genomics data and GWAS data [@doi:10.1038/s41588-018-0081-4; @doi:10.1016/j.ajhg.2018.04.002; @doi:10.1038/s41588-018-0081-4; @doi:10.1038/ncomms6890] has improved the identification of these transcriptional mechanisms that, when dysregulated, commonly result in tissue- and cell lineage-specific pathology [@pmid:20624743; @pmid:14707169; @doi:10.1073/pnas.0810772105].
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 5

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
        paragraph, model, "introduction"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[@" in paragraph_revised
    assert paragraph_revised.count("@") >= int(paragraph_text.count("@") * 0.50)

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_introduction_paragraph_with_citations_and_paragraph_is_the_last(model):
    # from LLM for articles revision manuscript
    paragraph = r"""
We developed a software publishing platform that imagines a future where authors co-write their manuscripts with the support of large language models.
We used, as a base, the Manubot platform for scholarly publishing [@doi:10.1371/journal.pcbi.1007128].
Manubot was designed as an end-to-end publishing platform for scholarly writing for both individual and large-collaborative projects.
It has been used for collaborations of approximately 50 authors writing hundreds of pages of text reviewing progress during the COVID19 pandemic [@pmid:34545336].
We developed a new workflow that parses the manuscript, uses a large language model with section-specific custom prompts to revise the manuscript, and then creates a set of suggested changes to reach the revised state.
Changes are presented to the user through the GitHub interface for author review and integration into the published document.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 6

    model.title = "A publishing infrastructure for AI-assisted academic authoring"
    model.keywords = [
        "manubot",
        "artificial intelligence",
        "scholarly publishing",
        "software",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "introduction"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 25
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # most citations were kept in the revised text
    assert "[@" in paragraph_revised
    assert paragraph_revised.count("@") >= int(paragraph_text.count("@") * 0.50)

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_results_paragraph_with_short_inline_formulas_and_refs_to_figures_and_citations(
    model,
):
    # from CCC manuscript
    paragraph = r"""
We examined how the Pearson ($p$), Spearman ($s$) and CCC ($c$) correlation coefficients behaved on different simulated data patterns.
In the first row of Figure @fig:datasets_rel, we examine the classic Anscombe's quartet [@doi:10.1080/00031305.1973.10478966], which comprises four synthetic datasets with different patterns but the same data statistics (mean, standard deviation and Pearson's correlation).
This kind of simulated data, recently revisited with the "Datasaurus" [@url:http://www.thefunctionalart.com/2016/08/download-datasaurus-never-trust-summary.html; @doi:10.1145/3025453.3025912; @doi:10.1111/dsji.12233], is used as a reminder of the importance of going beyond simple statistics, where either undesirable patterns (such as outliers) or desirable ones (such as biologically meaningful nonlinear relationships) can be masked by summary statistics alone.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 3

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "results"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # some citations were kept in the revised text
    assert "[@" in paragraph_revised

    # references to figures were kept
    assert "Figure @fig:datasets_rel" in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_results_paragraph_with_lists_and_refs_to_sections_and_subfigs(model):
    # from PhenoPLIER manuscript
    paragraph = r"""
PhenoPLIER is a flexible computational framework that combines gene-trait and gene-drug associations with gene modules expressed in specific contexts (Figure {@fig:entire_process}a).
The approach uses a latent representation (with latent variables or LVs representing gene modules) derived from a large gene expression compendium (Figure {@fig:entire_process}b, top) to integrate TWAS with drug-induced transcriptional responses (Figure {@fig:entire_process}b, bottom) for a joint analysis.
The approach consists in three main components (Figure {@fig:entire_process}b, middle, see [Methods](#sec:methods)):
1) an LV-based regression model to compute an association between an LV and a trait,
2) a clustering framework to learn groups of traits with shared transcriptomic properties,
and 3) an LV-based drug repurposing approach that links diseases to potential treatments.
We performed extensive simulations for our regression model ([Supplementary Note 1](#sm:reg:null_sim)) and clustering framework ([Supplementary Note 2](#sm:clustering:null_sim)) to ensure proper calibration and expected results under a model of no association.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 7

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
        paragraph, model, "results"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # list was kept
    assert "1)" in paragraph_revised
    assert "2)" in paragraph_revised
    assert "3)" in paragraph_revised

    # references to sub figures were kept
    assert "Figure {@fig:entire_process}a" in paragraph_revised
    assert "Figure {@fig:entire_process}b" in paragraph_revised

    # ferences to sections were kept
    assert "[Supplementary Note 1](#sm:reg:null_sim)" in paragraph_revised
    assert "[Supplementary Note 2](#sm:clustering:null_sim)" in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_results_paragraph_is_too_long(model):
    # from CCC manuscript
    paragraph = r"""
We sought to systematically analyze discrepant scores to assess whether associations were replicated in other datasets besides GTEx.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 1

    paragraph = paragraph * 200

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "results"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text

    # if there is an error, it should return the original paragraph with a header specifying the error
    error_message = r"""
<!--
ERROR: the paragraph below could not be revised with the AI model due to the following error:

This model's maximum context length is 4097 tokens, however you requested 17570 tokens (4272 in your prompt; 13298 for the completion). Please reduce your prompt; or completion length.
-->
    """.strip()
    assert starts_with_similar(
        paragraph_revised, error_message, 0.55 if not model.edit_endpoint else 0.30
    )

    # remove the multiline html comment at the top of the revised paragraph
    paragraph_revised_without_error = paragraph_revised.split("-->\n")[1].strip()
    assert "\n".join(paragraph) == paragraph_revised_without_error


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_discussion_paragraph_with_markdown_formatting_and_citations(model):
    # from CCC manuscript
    paragraph = r"""
It is well-known that biomedical research is biased towards a small fraction of human genes [@pmid:17620606; @pmid:17472739].
Some genes highlighted in CCC-ranked pairs (Figure @fig:upsetplot_coefs b), such as *SDS* (12q24) and *ZDHHC12* (9q34), were previously found to be the focus of fewer than expected publications [@pmid:30226837].
It is possible that the widespread use of linear coefficients may bias researchers away from genes with complex coexpression patterns.
A beyond-linear gene co-expression analysis on large compendia might shed light on the function of understudied genes.
For example, gene *KLHL21* (1p36) and *AC068580.6* (*ENSG00000235027*, in 11p15) have a high CCC value and are missed by the other coefficients.
*KLHL21* was suggested as a potential therapeutic target for hepatocellular carcinoma [@pmid:27769251] and other cancers [@pmid:29574153; @pmid:35084622].
Its nonlinear correlation with *AC068580.6* might unveil other important players in cancer initiation or progression, potentially in subsets of samples with specific characteristics (as suggested in Figure @fig:upsetplot_coefs b).
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 7

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "discussion"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # some citations were kept in the revised text
    assert "[@" in paragraph_revised

    # Markdown formatting was kept in the revised text
    assert "*" in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_discussion_paragraph_with_minor_math_and_refs_to_sections_and_websites(
    model,
):
    # from PhenoPLIER manuscript
    paragraph = r"""
Finally, we developed an LV-based regression framework to detect whether gene modules are associated with a trait using TWAS $p$-values.
We used PhenomeXcan as a discovery cohort across four thousand traits, and many LV-trait associations replicated in eMERGE.
In PhenomeXcan, we found 3,450 significant LV-trait associations (FDR < 0.05) with 686 LVs (out of 987) associated with at least one trait and 1,176 traits associated with at least one LV.
In eMERGE, we found 196 significant LV-trait associations, with 116 LVs associated with at least one trait/phecode and 81 traits with at least one LV.
We only focused on a few disease types from our trait clusters, but the complete set of associations on other disease domains is available in our [Github repository](https://github.com/greenelab/phenoplier) for future research.
As noted in [Methods](#sec:methods:reg), one limitation of the regression approach is that the gene-gene correlations are only approximately accurate, which could lead to false positives if the correlation among the top genes in a module is not precisely captured.
The regression model, however, is approximately well-calibrated, and we did not observe inflation when running the method in real data.
        """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 7

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
        paragraph, model, "discussion"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # equations or minor math were kept in the revised text
    # assert "$" in paragraph_revised
    assert "FDR < 0.05" in paragraph_revised

    # refs to external websites
    assert (
        "[Github repository](https://github.com/greenelab/phenoplier)"
        in paragraph_revised
    )


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_conclusions_paragraph_with_simple_text(model):
    # conclusions is the same as discussion in CCC/PhenoPLIER

    # from LLM for articles revision manuscript
    paragraph = r"""
We implemented AI-based models into publishing infrastructure.
While most manuscripts have been written by humans, the process is time consuming and academic writing can be difficult to parse.
We sought to develop a technology that academics could use to make their writing more understandable without changing the fundamental meaning.
This work lays the foundation for a future where academic manuscripts are constructed by a process that incorporates both human and machine authors.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 4

    model.title = "A publishing infrastructure for AI-assisted academic authoring"
    model.keywords = [
        "manubot",
        "artificial intelligence",
        "scholarly publishing",
        "software",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "conclusions"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # revised text does not have math or references
    assert "$" not in paragraph_revised
    assert "[" not in paragraph_revised
    assert "@" not in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_methods_paragraph_with_inline_equations_and_figure_refs(model):
    # from CCC manuscript
    paragraph = r"""
The Clustermatch Correlation Coefficient (CCC) computes a similarity value $c \in \left[0,1\right]$ between any pair of numerical or categorical features/variables $\mathbf{x}$ and $\mathbf{y}$ measured on $n$ objects.
CCC assumes that if two features $\mathbf{x}$ and $\mathbf{y}$ are similar, then the partitioning by clustering of the $n$ objects using each feature separately should match.
For example, given $\mathbf{x}=(11, 27, 32, 40)$ and $\mathbf{y}=10x=(110, 270, 320, 400)$, where $n=4$, partitioning each variable into two clusters ($k=2$) using their medians (29.5 for $\mathbf{x}$ and 295 for $\mathbf{y}$) would result in partition $\Omega^{\mathbf{x}}_{k=2}=(1, 1, 2, 2)$ for $\mathbf{x}$, and partition $\Omega^{\mathbf{y}}_{k=2}=(1, 1, 2, 2)$ for $\mathbf{y}$.
Then, the agreement between $\Omega^{\mathbf{x}}_{k=2}$ and $\Omega^{\mathbf{y}}_{k=2}$ can be computed using any measure of similarity between partitions, like the adjusted Rand index (ARI) [@doi:10.1007/BF01908075].
In that case, it will return the maximum value (1.0 in the case of ARI).
Note that the same value of $k$ might not be the right one to find a relationship between any two features.
For instance, in the quadratic example in Figure @fig:datasets_rel, CCC returns a value of 0.36 (grouping objects in four clusters using one feature and two using the other).
If we used only two clusters instead, CCC would return a similarity value of 0.02.
Therefore, the CCC algorithm (shown below) searches for this optimal number of clusters given a maximum $k$, which is its single parameter $k_{\mathrm{max}}$.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 9

    model.title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    model.keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # some formulas are referenced in the revised text
    assert "$" in paragraph_revised

    # some figures are referenced in the revised text
    assert "Figure @fig:datasets_rel" in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_methods_paragraph_with_figure_table_and_equation_refs(model):
    # from PhenoPLIER manuscript:
    paragraph = r"""
Note that, since we used the MultiXcan regression model (Equation (@eq:multixcan)), $\mathbf{R}$ is only an approximation of gene correlations in S-MultiXcan.
As explained before, S-MultiXcan approximates the joint regression parameters in MultiXcan using the marginal regression estimates from S-PrediXcan in (@eq:spredixcan) with some simplifying assumptions and different genotype covariance matrices.
This complicates the derivation of an S-MultiXcan-specific solution to compute $\mathbf{R}$.
To account for this, we used a submatrix $\mathbf{R}_{\ell}$ corresponding to genes that are part of LV $\ell$ only (top 1% of genes) instead of the entire matrix $\mathbf{R}$.
This simplification is conservative since correlations are accounted for top genes only.
Our simulations ([Supplementary Note 1](#sm:reg:null_sim)) show that the model is approximately well-calibrated and can correct for LVs with adjacent and highly correlated genes at the top (e.g., Figure @fig:reg:nulls:qqplot:lv234).
The model can also detect LVs associated with relevant traits (Figure @fig:lv246 and Table @tbl:sup:phenomexcan_assocs:lv246) that are replicated in a different cohort (Table @tbl:sup:emerge_assocs:lv246).
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 7

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
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # some equations are referenced in the revised text
    assert ("Equation (@eq:multixcan)" in paragraph_revised) or (
        "Equation (@eq:spredixcan)" in paragraph_revised
    )

    # some figures/tables are referenced in the revised text
    assert "Figure @fig:lv246" in paragraph_revised
    assert "Table @tbl:sup:phenomexcan_assocs:lv246" in paragraph_revised
    assert "Table @tbl:sup:emerge_assocs:lv246" in paragraph_revised

    # reference to important sections
    assert "[Supplementary Note 1](#sm:reg:null_sim)" in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_methods_paragraph_with_inline_math_and_equations(model):
    # from PhenoPLIER manuscript:
    paragraph = r"""
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
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 18

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
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    # some equations are referenced in the revised text
    assert "$$ {#eq:predixcan}" in paragraph_revised
    assert "$$ {#eq:spredixcan}" in paragraph_revised
    assert "$\hat{\sigma}_a$" in paragraph_revised


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_methods_paragraph_without_fig_table_reference(model):
    # from LLM for articles revision manuscript
    paragraph = r"""
We used the OpenAI API for access to large language models, with a focus on the completion endpoints.
This API incurs a cost with each run that depends on manuscript length.
Because of this cost, we implemented our workflow in GitHub actions, making it triggerable by the user.
The user can select the model that they wish to use, allowing costs to be tuned.
With the most complex model, `text-davinci-003`, the cost per run is under $0.50 for many manuscripts.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 5

    model.title = "A publishing infrastructure for AI-assisted academic authoring"
    model.keywords = [
        "manubot",
        "artificial intelligence",
        "scholarly publishing",
        "software",
    ]

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert paragraph_revised[-1] == "."

    assert "`text-davinci-003`" in paragraph_revised

    # no figures/tables are referenced in the revised text
    assert "figure" not in paragraph_revised.lower()
    assert "table" not in paragraph_revised.lower()
    assert "@" not in paragraph_revised.lower()


@pytest.mark.parametrize(
    "model",
    [
        GPT3CompletionModel(None, None),
        GPT3CompletionModel(None, None, model_engine="text-davinci-edit-001"),
        GPT3CompletionModel(None, None, model_engine="gpt-3.5-turbo"),
    ],
)
@pytest.mark.cost
def test_revise_methods_paragraph_with_many_tokens(model):
    # from PhenoPLIER manuscript:
    paragraph = r"""
Since the error terms $\bm{\epsilon}$ could be correlated, we cannot assume they have independent normal distributions as in a standard linear regression model.
In the PrediXcan family of methods, the predicted expression of a pair of genes could be correlated if they share eQTLs or if these are in LD [@doi:10.1038/s41588-019-0385-z].
Therefore, we used a generalized least squares approach to account for these correlations.
The gene-gene correlation matrix $\mathbf{R}$ was approximated by computing the correlations between the model sum of squares (SSM) for each pair of genes under the null hypothesis of no association.
These correlations are derived from the individual-level MultiXcan model (Equation (@eq:multixcan)), where the predicted expression matrix $\mathbf{T}_{i} \in \mathbb{R}^{n \times p_i}$ of a gene $i$ across $p_i$ tissues is projected into its top $k_i$ PCs, resulting in matrix $\mathbf{P}_{i} \in \mathbb{R}^{n \times k_i}$.
From the MAGMA framework, we know that the SSM for each gene is proportial to $\mathbf{y}^{\top} \mathbf{P}_{i} \mathbf{P}_{i}^{\top} \mathbf{y}$.
Under the null hypothesis of no association, the covariances between the SSM of genes $i$ and $j$ is therefore given by $2 \times \mathrm{Trace}(\mathbf{P}_{i}^{\top} \mathbf{P}_{j} \mathbf{P}_{j}^{\top} \mathbf{P}_{i})$.
The standard deviations of each SSM are given by $\sqrt{2 \times k_{i}} \times (n - 1)$.
Therefore, the correlation between the SSMs for genes $i$ and $j$ can be written as follows:

$$
\begin{split}
\mathbf{R}_{ij} & = \frac{2 \times \mathrm{Tr}(\mathbf{P}_{i}^{\top} \mathbf{P}_{j} \mathbf{P}_{j}^{\top} \mathbf{P}_{i})}{\sqrt{2 \times k_{i}} \times \sqrt{2 \times k_{j}} \times (n - 1)^2} \\
& = \frac{2 \times \mathrm{Tr}(Cor(\mathbf{P}_{i}, \mathbf{P}_{j}) \times Cor(\mathbf{P}_{j}, \mathbf{P}_{i}))}{\sqrt{2 \times k_{i}} \times \sqrt{2 \times k_{j}}},
\end{split}
$$ {#eq:reg:r}

where columns $\mathbf{P}$ are standardized,
$\mathrm{Tr}$ is the trace of a matrix,
and the cross-correlation matrix between PCs $Cor(\mathbf{P}_{i}, \mathbf{P}_{j}) \in \mathbb{R}^{k_i \times k_j}$ is given by

$$
\begin{split}
Cor(\mathbf{P}_{i}, \mathbf{P}_{j}) & = Cor(\mathbf{T}_{i} \mathbf{V}_{i}^{\top} \mathrm{diag}(\lambda_i)^{-1/2}, \mathbf{T}_{j} \mathbf{V}_{j}^{\top} \mathrm{diag}(\lambda_j)^{-1/2}) \\
& = \mathrm{diag}(\lambda_i)^{-1/2} \mathbf{V}_{i} (\frac{\mathbf{T}_{i}^{\top} \mathbf{T}_{j}}{n-1}) \mathbf{V}_{j}^{\top} \mathrm{diag}(\lambda_j)^{-1/2},
\end{split}
$$ {#eq:reg:cor_pp}

where $\frac{\mathbf{T}_{i}^{\top} \mathbf{T}_{j}}{n-1} \in \mathbb{R}^{p_i \times p_j}$ is the cross-correlation matrix between the predicted expression levels of genes $i$ and $j$,
and columns of $\mathbf{V}_{i}$ and scalars $\lambda_i$ are the eigenvectors and eigenvalues of $\mathbf{T}_{i}$, respectively.
S-MultiXcan keeps only the top eigenvectors using a condition number threshold of $\frac{\max(\lambda_i)}{\lambda_i} < 30$.
To estimate the correlation of predicted expression levels for genes $i$ in tissue $k$ and gene $j$ in tissue $l$, $(\mathbf{t}_k^i, \mathbf{t}_l^j)$ ($\mathbf{t}_k^i$ is the $k$th column of $\mathbf{T}_{i}$), we used [@doi:10.1371/journal.pgen.1007889]

$$
\begin{split}
\frac{(\mathbf{T}_{i}^{\top} \mathbf{T}_{j})_{kl}}{n-1} & = Cor(\mathbf{t}_k^i, \mathbf{t}_l^j) \\
 & = \frac{ Cov(\mathbf{t}_k, \mathbf{t}_l) } { \sqrt{\widehat{\mathrm{var}}(\mathbf{t}_k) \widehat{\mathrm{var}}(\mathbf{t}_l)} } \\
 & = \frac{ Cov(\sum_{a \in \mathrm{model}_k} w_a^k X_a, \sum_{b \in \mathrm{model}_l} w_b^l X_b) }  {\sqrt{\widehat{\mathrm{var}}(\mathbf{t}_k) \widehat{\mathrm{var}}(\mathbf{t}_l)} } \\
 & = \frac{ \sum_{a \in \mathrm{model}_k \\ b \in \mathrm{model}_l} w_a^k w_b^l Cov(X_a, X_b)} {\sqrt{\widehat{\mathrm{var}}(\mathbf{t}_k) \widehat{\mathrm{var}}(\mathbf{t}_l)} } \\
 & = \frac{ \sum_{a \in \mathrm{model}_k \\ b \in \mathrm{model}_l} w_a^k w_b^l \Gamma_{ab}} {\sqrt{\widehat{\mathrm{var}}(\mathbf{t}_k) \widehat{\mathrm{var}}(\mathbf{t}_l)} },
\end{split}
$$ {#eq:reg:corr_genes}

where $X_a$ is the genotype of SNP $a$,
$w_a^k$ is the weight of SNP $a$ for gene expression prediction in the tissue model $k$,
and $\Gamma = \widehat{\mathrm{var}}(\mathbf{X}) = (\mathbf{X} - \mathbf{\bar{X}})^{\top} (\mathbf{X} - \mathbf{\bar{X}}) / (n-1)$ is the genotype covariance matrix using GTEx v8 as the reference panel, which is the same used in all TWAS methods described here.
The variance of the predicted expression values of gene $i$ in tissue $k$ is estimated as [@doi:10.1038/s41467-018-03621-1]:

$$
\begin{split}
\widehat{\mathrm{var}}(\mathbf{t}_k^i) & = (\mathbf{W}^k)^\top \Gamma^k \mathbf{W}^k \\
 & = \sum_{a \in \mathrm{model}_k \\ b \in \mathrm{model}_k} w_a^k w_b^k \Gamma_{ab}^k.
\end{split}
$$ {#eq:reg:var_gene}
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 54

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
        paragraph, model, "methods"
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100
    assert "<!--\nERROR:" not in paragraph_revised

    # revised paragraph was finished (no incomplete sentences, which could happen
    # if the max_tokens parameter is too low)
    assert (paragraph_revised[-1] == ".") or (paragraph_revised[-1] == "}")

    # some equations are referenced in the revised text
    assert "$$ {#eq:reg:r}" in paragraph_revised
    assert "$Cor(\mathbf{P}_{i}, \mathbf{P}_{j})" in paragraph_revised
