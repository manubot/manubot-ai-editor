from chatgpt_editor.models import GPT3CompletionModel


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
    paragraph_text = """
Correlation coefficients are widely used to identify patterns in data that may be of particular interest.
In transcriptomics, genes with correlated expression often share functions or are part of disease-relevant biological processes.
Here we introduce the Clustermatch Correlation Coefficient (CCC), an efficient, easy-to-use and not-only-linear coefficient based on machine learning models.
CCC reveals biologically meaningful linear and nonlinear patterns missed by standard, linear-only correlation coefficients.
CCC captures general patterns in data by comparing clustering solutions while being much faster than state-of-the-art coefficients such as the Maximal Information Coefficient.
When applied to human gene expression data, CCC identifies robust linear relationships while detecting nonlinear patterns associated, for example, with sex differences that are not captured by linear-only coefficients.
Gene pairs highly ranked by CCC were enriched for interactions in integrated networks built from protein-protein interaction, transcription factor regulation, and chemical and genetic perturbations, suggesting that CCC could detect functional relationships that linear-only methods missed.
CCC is a highly-efficient, next-generation not-only-linear correlation coefficient that can readily be applied to genome-scale data and other domains across different data types.
    """.strip()

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_revised = model.revise_paragraph(paragraph_text, "abstract")
    assert paragraph_revised is not None
    assert isinstance(paragraph_revised, str)
    assert paragraph_revised != paragraph_text


def test_revise_introduction_paragraph():
    paragraph_text = """
New technologies have vastly improved data collection, generating a deluge of information across different disciplines.
This large amount of data provides new opportunities to address unanswered scientific questions, provided we have efficient tools capable of identifying multiple types of underlying patterns.
Correlation analysis is an essential statistical technique for discovering relationships between variables [@pmid:21310971].
Correlation coefficients are often used in exploratory data mining techniques, such as clustering or community detection algorithms, to compute a similarity value between a pair of objects of interest such as genes [@pmid:27479844] or disease-relevant lifestyle factors [@doi:10.1073/pnas.1217269109].
Correlation methods are also used in supervised tasks, for example, for feature selection to improve prediction accuracy [@pmid:27006077; @pmid:33729976].
The Pearson correlation coefficient is ubiquitously deployed across application domains and diverse scientific areas.
Thus, even minor and significant improvements in these techniques could have enormous consequences in industry and research.
    """.strip()

    model = GPT3CompletionModel(
        title="An efficient not-only-linear correlation coefficient based on machine learning",
        keywords=[
            "correlation coefficient",
            "nonlinear relationships",
            "gene expression",
        ],
    )

    paragraph_revised = model.revise_paragraph(paragraph_text, "introduction")
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
