from chatgpt_editor.editor import ManuscriptEditor
from chatgpt_editor.models import GPT3CompletionModel
from chatgpt_editor.utils import SENTENCE_END_PATTERN


def test_model_object_init():
    model = GPT3CompletionModel(
        title="Test title",
        keywords=["test", "keywords"],
    )


def test_get_prompt_for_abstract():
    paragraph_text = "Text of the abstract"
    title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    model = GPT3CompletionModel(
        title=title,
        keywords=keywords,
    )

    prompt = model.get_prompt(paragraph_text, "abstract")
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "abstract" in prompt
    assert f"'{title}'" in prompt
    assert f"{keywords[0]}" in prompt
    assert f"{keywords[1]}" in prompt
    assert f"{keywords[2]}" in prompt
    assert paragraph_text in prompt
    assert prompt.startswith("Revise")
    assert "  " not in prompt


def test_get_prompt_for_introduction():
    paragraph_text = "Text of the initial part."
    title = (
        "An efficient not-only-linear correlation coefficient based on machine learning"
    )
    keywords = [
        "correlation coefficient",
        "nonlinear relationships",
        "gene expression",
    ]

    model = GPT3CompletionModel(
        title=title,
        keywords=keywords,
    )

    prompt = model.get_prompt(paragraph_text, "introduction")
    assert prompt is not None
    assert isinstance(prompt, str)
    assert "introduction" in prompt
    assert f"'{title}'" in prompt
    assert f"{keywords[0]}" in prompt
    assert f"{keywords[1]}" in prompt
    assert f"{keywords[2]}" in prompt
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
    assert len(paragraph_revised) > 10


def test_revise_introduction_paragraph_with_citations():
    paragraph = """
New technologies have vastly improved data collection, generating a deluge of information across different disciplines.
This large amount of data provides new opportunities to address unanswered scientific questions, provided we have efficient tools capable of identifying multiple types of underlying patterns.
Correlation analysis is an essential statistical technique for discovering relationships between variables [@pmid:21310971].
Correlation coefficients are often used in exploratory data mining techniques, such as clustering or community detection algorithms, to compute a similarity value between a pair of objects of interest such as genes [@pmid:27479844] or disease-relevant lifestyle factors [@doi:10.1073/pnas.1217269109].
Correlation methods are also used in supervised tasks, for example, for feature selection to improve prediction accuracy [@pmid:27006077; @pmid:33729976].
The Pearson correlation coefficient is ubiquitously deployed across application domains and diverse scientific areas.
Thus, even minor and significant improvements in these techniques could have enormous consequences in industry and research.
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

    # make sure manubot references were kept
    assert "[@" in paragraph_revised
    assert "@pmid:21310971" in paragraph_revised
    assert "@pmid:27479844" in paragraph_revised
    assert "@pmid:27006077" in paragraph_revised
    assert "@pmid:33729976" in paragraph_revised
    assert "@doi:10.1073/pnas.1217269109" in paragraph_revised


def test_revise_results_paragraph_with_citations():
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

    # make sure manubot references were kept
    assert "[@" in paragraph_revised
    assert "@doi:10.1080/00031305.1973.10478966" in paragraph_revised
    assert (
        "@url:http://www.thefunctionalart.com/2016/08/download-datasaurus-never-trust-summary.html"
        in paragraph_revised
    )
    assert "@doi:10.1145/3025453.3025912" in paragraph_revised
    assert "@doi:10.1111/dsji.12233" in paragraph_revised


def test_revise_results_paragraph_with_figure_ref():
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

    # make sure manubot references were kept
    assert "@fig:datasets_rel" in paragraph_revised


def test_revise_too_long_paragraph():
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

    # if there is an error, it should return the original paragraph
    assert SENTENCE_END_PATTERN.sub(".\n", paragraph_text) == paragraph_revised


def test_revise_discussion_paragraph():
    paragraph = """
We introduce the Clustermatch Correlation Coefficient (CCC), an efficient not-only-linear machine learning-based statistic.
Applying CCC to GTEx v8 revealed that it was robust to outliers and detected linear relationships as well as complex and biologically meaningful patterns that standard coefficients missed.
In particular, CCC alone detected gene pairs with complex nonlinear patterns from the sex chromosomes, highlighting the way that not-only-linear coefficients can play in capturing sex-specific differences.
The ability to capture these nonlinear patterns, however, extends beyond sex differences: it provides a powerful approach to detect complex relationships where a subset of samples or conditions are explained by other factors (such as differences between health and disease).
We found that top CCC-ranked gene pairs in whole blood from GTEx were replicated in independent tissue-specific networks trained from multiple data types and attributed to cell lineages from blood, even though CCC did not have access to any cell lineage-specific information.
This suggests that CCC can disentangle intricate cell lineage-specific transcriptional patterns missed by linear-only coefficients.
In addition to capturing nonlinear patterns, the CCC was more similar to Spearman than Pearson, highlighting their shared robustness to outliers.
The CCC results were concordant with MIC, but much faster to compute and thus practical for large datasets.
Another advantage over MIC is that CCC can also process categorical variables together with numerical values.
CCC is conceptually easy to interpret and has a single parameter that controls the maximum complexity of the detected relationships while also balancing compute time.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 10

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


def test_revise_methods_paragraph():
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

    assert "$" in paragraph_revised


def test_revise_supplementary_material_paragraph():
    paragraph = """
We compared all the coefficients in this study with MIC [@pmid:22174245], a popular nonlinear method that can find complex relationships in data, although very computationally intensive [@doi:10.1098/rsos.201424].
We ran MIC<sub>e</sub> (see Methods) on all possible pairwise comparisons of our 5,000 highly variable genes from whole blood in GTEx v8.
This took 4 days and 19 hours to finish (compared with 9 hours for CCC).
Then we performed the analysis on the distribution of coefficients (the same as in the main text), shown in Figure @fig:dist_coefs_mic.
We verified that CCC and MIC behave similarly in this dataset, with essentially the same distribution but only shifted.
Figure @fig:dist_coefs_mic c shows that these two coefficients relate almost linearly, and both compare very similarly with Pearson and Spearman.
    """.strip().split(
        "\n"
    )
    paragraph = [sentence.strip() for sentence in paragraph]
    assert len(paragraph) == 6

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_text, paragraph_revised = ManuscriptEditor.revise_and_write_paragraph(
        paragraph, "supplementary_material", model
    )
    assert paragraph_text is not None
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text
    assert len(paragraph_revised) > 100

    assert "@fig:dist_coefs_mic" in paragraph_revised
