import os
from pathlib import Path
from unittest import mock

import pytest

from manubot.ai_editor.editor import ManuscriptEditor
from manubot.ai_editor import env_vars, models
from manubot.ai_editor.models import GPT3CompletionModel
from manubot.ai_editor.utils import SENTENCE_END_PATTERN


MANUSCRIPTS_DIR = Path(__file__).parent / "manuscripts"


def test_model_object_init_without_openai_api_key():
    _environ = os.environ.copy()
    try:
        if env_vars.OPENAI_API_KEY in os.environ:
            os.environ.pop(env_vars.OPENAI_API_KEY)

        with pytest.raises(ValueError):
            GPT3CompletionModel(
                title="Test title",
                keywords=["test", "keywords"],
            )
    finally:
        os.environ = _environ


@mock.patch.dict("os.environ", {env_vars.OPENAI_API_KEY: "env_var_test_value"})
def test_model_object_init_with_openai_api_key_as_environment_variable():
    GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert models.openai.api_key == "env_var_test_value"


def test_model_object_init_with_openai_api_key_as_parameter():
    _environ = os.environ.copy()
    try:
        if env_vars.OPENAI_API_KEY in os.environ:
            os.environ.pop(env_vars.OPENAI_API_KEY)

        GPT3CompletionModel(
            title="Test title",
            keywords=["test", "keywords"],
            openai_api_key="test_value",
        )

        from manubot.ai_editor import models

        assert models.openai.api_key == "test_value"
    finally:
        os.environ = _environ


@mock.patch.dict("os.environ", {env_vars.OPENAI_API_KEY: "env_var_test_value"})
def test_model_object_init_with_openai_api_key_as_parameter_has_higher_priority():
    GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
        openai_api_key="test_value",
    )

    from manubot.ai_editor import models

    assert models.openai.api_key == "test_value"


def test_model_object_init_default_language_model():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert model.model_parameters["engine"] == "text-davinci-003"


@mock.patch.dict("os.environ", {env_vars.LANGUAGE_MODEL: "text-curie-001"})
def test_model_object_init_read_language_model_from_environment():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )

    assert model.model_parameters["engine"] == "text-curie-001"


def test_get_prompt_for_abstract():
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    paragraph_text = "Text of the abstract"

    prompt = model.get_prompt(paragraph_text, "abstract")
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "abstract" in prompt
    assert f"'{me.title}'" in prompt
    assert f"{me.keywords[0]}" in prompt
    assert f"{me.keywords[1]}" in prompt
    assert f"{me.keywords[2]}" in prompt
    assert paragraph_text in prompt
    assert prompt.startswith("Revise")
    assert "  " not in prompt


def test_get_prompt_for_introduction():
    me = ManuscriptEditor(
        content_dir=MANUSCRIPTS_DIR / "ccc",
    )

    model = GPT3CompletionModel(
        title=me.title,
        keywords=me.keywords,
    )

    paragraph_text = "Text of the initial part"

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "introduction" in prompt
    assert f"'{me.title}'" in prompt
    assert f"{me.keywords[0]}" in prompt
    assert f"{me.keywords[1]}" in prompt
    assert f"{me.keywords[2]}" in prompt
    assert paragraph_text in prompt
    assert prompt.startswith("Revise")
    assert "  " not in prompt


def test_revise_abstract():
    paragraph = """
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

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "abstract", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text


def test_revise_introduction_paragraph_with_single_and_multiple_citations_together():
    # from CCC manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "introduction", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # most citations were kept in the revised text
    assert "[@" in paragraph_revised
    assert paragraph_revised.count("@") >= int(paragraph_text.count("@") * 0.75)

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


def test_revise_introduction_paragraph_with_citations_and_paragraph_is_the_first():
    # from PhenoPLIER manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="Projecting genetic associations through gene expression patterns highlights disease etiology and drug mechanisms",
        keywords=[
            "genetic studies",
            "functional genomics",
            "gene co-expression",
            "gene prioritization",
            "drug repurposing",
            "clustering of complex traits",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "introduction", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # most citations were kept in the revised text
    assert "[@" in paragraph_revised
    assert paragraph_revised.count("@") >= int(paragraph_text.count("@") * 0.75)

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


def test_revise_introduction_paragraph_with_citations_and_paragraph_is_the_last():
    # from LLM for articles revision manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="A publishing infrastructure for AI-assisted academic authoring",
        keywords=[
            "manubot",
            "artificial intelligence",
            "scholarly publishing",
            "software",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "introduction", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # most citations were kept in the revised text
    assert "[@" in paragraph_revised
    assert paragraph_revised.count("@") >= int(paragraph_text.count("@") * 0.75)

    # no references to figures or tables
    assert "Figure" not in paragraph_revised
    assert "Table" not in paragraph_revised

    # no math
    assert "$" not in paragraph_revised


def test_revise_results_paragraph_with_short_inline_formulas_and_refs_to_figures_and_citations():
    # from CCC manuscript
    paragraph = """
We examined how the Pearson ($p$), Spearman ($s$) and CCC ($c$) correlation coefficients behaved on different simulated data patterns.
In the first row of Figure @fig:datasets_rel, we examine the classic Anscombe's quartet [@doi:10.1080/00031305.1973.10478966], which comprises four synthetic datasets with different patterns but the same data statistics (mean, standard deviation and Pearson's correlation).
This kind of simulated data, recently revisited with the "Datasaurus" [@url:http://www.thefunctionalart.com/2016/08/download-datasaurus-never-trust-summary.html; @doi:10.1145/3025453.3025912; @doi:10.1111/dsji.12233], is used as a reminder of the importance of going beyond simple statistics, where either undesirable patterns (such as outliers) or desirable ones (such as biologically meaningful nonlinear relationships) can be masked by summary statistics alone.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 3

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "results", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # some citations were kept in the revised text
    assert "[@" in paragraph_revised

    # references to figures were kept
    assert "Figure @fig:datasets_rel" in paragraph_revised


def test_revise_results_paragraph_with_lists_and_refs_to_sections_and_subfigs():
    # from PhenoPLIER manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="Projecting genetic associations through gene expression patterns highlights disease etiology and drug mechanisms",
        keywords=[
            "gene co-expression",
            "MultiPLIER",
            "PhenomeXcan",
            "TWAS",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "results", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

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


def test_revise_results_paragraph_is_too_long():
    # from CCC manuscript
    paragraph = """
We sought to systematically analyze discrepant scores to assess whether associations were replicated in other datasets besides GTEx.
This is challenging and prone to bias because linear-only correlation coefficients are usually used in gene co-expression analyses.
We used 144 tissue-specific gene networks from the Genome-wide Analysis of gene Networks in Tissues (GIANT) [@pmcid:PMC4828725; @url:https://hb.flatironinstitute.org], where nodes represent genes and each edge a functional relationship weighted with a probability of interaction between two genes (see [Methods](#sec:giant)).
Importantly, the version of GIANT used in this study did not include GTEx samples [@url:https://hb.flatironinstitute.org/data], making it an ideal case for replication.
These networks were built from expression and different interaction measurements, including protein-interaction, transcription factor regulation, chemical/genetic perturbations and microRNA target profiles from the Molecular Signatures Database (MSigDB [@pmid:16199517]).
We reasoned that highly-ranked gene pairs using three different coefficients in a single tissue (whole blood in GTEx, Figure @fig:upsetplot_coefs) that represented real patterns should often replicate in a corresponding tissue or related cell lineage using the multi-cell type functional interaction networks in GIANT.
In addition to predicting a network with interactions for a pair of genes, the GIANT web application can also automatically detect a relevant tissue or cell type where genes are predicted to be specifically expressed (the approach uses a machine learning method introduced in [@doi:10.1101/gr.155697.113] and described in [Methods](#sec:giant)).
For example, we obtained the networks in blood and the automatically-predicted cell type for gene pairs *RASSF2* - *CYTIP* (CCC high, Figure @fig:giant_gene_pairs a) and *MYOZ1* - *TNNI2* (Pearson high, Figure @fig:giant_gene_pairs b).
In addition to the gene pair, the networks include other genes connected according to their probability of interaction (up to 15 additional genes are shown), which allows estimating whether genes are part of the same tissue-specific biological process.
Two large black nodes in each network's top-left and bottom-right corners represent our gene pairs.
A green edge means a close-to-zero probability of interaction, whereas a red edge represents a strong predicted relationship between the two genes.
In this example, genes *RASSF2* and *CYTIP* (Figure @fig:giant_gene_pairs a), with a high CCC value ($c=0.20$, above the 73th percentile) and low Pearson and Spearman ($p=0.16$ and $s=0.11$, below the 38th and 17th percentiles, respectively), were both strongly connected to the blood network, with interaction scores of at least 0.63 and an average of 0.75 and 0.84, respectively (Supplementary Table @tbl:giant:weights).
The autodetected cell type for this pair was leukocytes, and interaction scores were similar to the blood network (Supplementary Table @tbl:giant:weights).
However, genes *MYOZ1* and *TNNI2*, with a very high Pearson value ($p=0.97$), moderate Spearman ($s=0.28$) and very low CCC ($c=0.03$), were predicted to belong to much less cohesive networks (Figure @fig:giant_gene_pairs b), with average interaction scores of 0.17 and 0.22 with the rest of the genes, respectively.
Additionally, the autodetected cell type (skeletal muscle) is not related to blood or one of its cell lineages.
These preliminary results suggested that CCC might be capturing blood-specific patterns missed by the other coefficients.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 16

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "results", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text

    # if there is an error, it should return the original paragraph with a header specifying the error
    error_message = r"""
<!--
ERROR: this paragraph could not be revised with the AI model due to the following error:

This model's maximum context length is 4097 tokens, however you requested 4498 tokens (934 in your prompt; 3564 for the completion). Please reduce your prompt; or completion length.
-->
    """.strip()
    assert paragraph_revised.startswith(error_message)

    paragraph_revised_without_error = paragraph_revised.replace(
        error_message + "\n", ""
    )
    assert (
        SENTENCE_END_PATTERN.sub(".\n", paragraph_text)
        == paragraph_revised_without_error
    )


def test_revise_discussion_paragraph_with_markdown_formatting_and_citations():
    # from CCC manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "discussion", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # some citations were kept in the revised text
    assert "[@" in paragraph_revised

    # Markdown formatting was kept in the revised text
    assert "*" in paragraph_revised


def test_revise_discussion_paragraph_with_minor_math_and_refs_to_sections_and_websites():
    # from PhenoPLIER manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="Projecting genetic associations through gene expression patterns highlights disease etiology and drug mechanisms",
        keywords=[
            "gene co-expression",
            "MultiPLIER",
            "PhenomeXcan",
            "TWAS",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "discussion", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # equations or minor math were kept in the revised text
    assert "$" in paragraph_revised
    assert "FDR < 0.05" in paragraph_revised

    # refs to external websites
    assert (
        "[Github repository](https://github.com/greenelab/phenoplier)"
        in paragraph_revised
    )


def test_revise_conclusions_paragraph_with_simple_text():
    # conclusions is the same as discussion in CCC/PhenoPLIER

    # from LLM for articles revision manuscript
    paragraph = """
We implemented AI-based models into publishing infrastructure.
While most manuscripts have been written by humans, the process is time consuming and academic writing can be difficult to parse.
We sought to develop a technology that academics could use to make their writing more understandable without changing the fundamental meaning.
This work lays the foundation for a future where academic manuscripts are constructed by a process that incorporates both human and machine authors.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 4

    model = GPT3CompletionModel(
        title="A publishing infrastructure for AI-assisted academic authoring",
        keywords=[
            "manubot",
            "artificial intelligence",
            "scholarly publishing",
            "software",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "conclusions", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # revised text does not have math or references
    assert "$" not in paragraph_revised
    assert "[" not in paragraph_revised
    assert "@" not in paragraph_revised


def test_revise_methods_paragraph_with_inline_equations_and_figure_refs():
    # from CCC manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "methods", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # some formulas are referenced in the revised text
    assert "$" in paragraph_revised

    # some figures are referenced in the revised text
    assert "Figure @fig:datasets_rel" in paragraph_revised


def test_revise_methods_paragraph_with_figure_table_and_equation_refs():
    # from PhenoPLIER manuscript:
    paragraph = """
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

    model = GPT3CompletionModel(
        title="Projecting genetic associations through gene expression patterns highlights disease etiology and drug mechanisms",
        keywords=[
            "gene co-expression",
            "MultiPLIER",
            "PhenomeXcan",
            "TWAS",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "methods", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    # some equations are referenced in the revised text
    assert ("Equation (@eq:multixcan)" in paragraph_revised) or (
        "Equation @eq:multixcan" in paragraph_revised
    )

    # some figures/tables are referenced in the revised text
    assert "Figure @fig:lv246" in paragraph_revised
    assert "Table @tbl:sup:phenomexcan_assocs:lv246" in paragraph_revised
    assert "Table @tbl:sup:emerge_assocs:lv246" in paragraph_revised

    # reference to important sections
    assert "[Supplementary Note 1](#sm:reg:null_sim)" in paragraph_revised


def test_revise_methods_paragraph_without_fig_table_reference():
    # from LLM for articles revision manuscript
    paragraph = """
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

    model = GPT3CompletionModel(
        title="A publishing infrastructure for AI-assisted academic authoring",
        keywords=[
            "manubot",
            "artificial intelligence",
            "scholarly publishing",
            "software",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "methods", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    assert "`text-davinci-003`" in paragraph_revised

    # no figures/tables are referenced in the revised text
    assert "figure" not in paragraph_revised.lower()
    assert "table" not in paragraph_revised.lower()
    assert "@" not in paragraph_revised.lower()
