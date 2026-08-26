import importlib


mod = importlib.import_module("294_open_ended_perturbation")


def test_trivia_competitor_excludes_pred_aliases():
    result = mod.choose_competitor(
        "trivia", "The Pilgrim's Progress",
        ["Pilgrims Progress", "Don Quixote", "Don Quixote", "Hamlet"])
    assert result["cluster_key"] == "don quixote"
    assert result["cluster_count"] == 2


def test_gsm8k_competitor_uses_final_number_cluster():
    result = mod.choose_competitor(
        "gsm8k", "work\n#### 11",
        ["different work\n#### 11", "x\n#### 12", "y\n#### 12", "#### 9"])
    assert result["cluster_key"] == "12"
    assert result["cluster_count"] == 2


def test_no_distinct_cluster_returns_none():
    assert mod.choose_competitor("gsm8k", "#### 4", ["#### 4", "answer is 4"]) is None
