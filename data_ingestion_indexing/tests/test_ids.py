# 中文功能说明：数据接入 ID 工具测试，验证 arXiv、标题和 chunk 标识生成逻辑。

from scholar_ingest.ids import normalize_arxiv_id, paper_id_from_arxiv, slugify_title


def test_normalize_arxiv_id_removes_url_and_version():
    assert normalize_arxiv_id("https://arxiv.org/abs/2309.04564v2") == "2309.04564"


def test_paper_id_from_arxiv():
    assert paper_id_from_arxiv("arXiv:2402.09668") == "arxiv:2402.09668"


def test_slugify_title_matches_pasa_zip_style():
    assert slugify_title("A Critique of Chen's \"The 2-MAXSAT Problem Can Be Solved in Polynomial Time\"") == (
        "acritiqueofchensthemaxsatproblemcanbesolvedinpolynomialtime"
    )


if __name__ == "__main__":
    test_normalize_arxiv_id_removes_url_and_version()
    test_paper_id_from_arxiv()
    test_slugify_title_matches_pasa_zip_style()
    print("ok")
