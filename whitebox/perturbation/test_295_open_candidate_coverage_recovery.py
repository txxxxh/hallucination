import importlib


mod = importlib.import_module("295_open_candidate_coverage_recovery")


def test_containment_alias_is_equivalent():
    assert mod.equivalent("trivia", "The Pilgrim's Progress",
                          "The Pilgrim's Progress by John Bunyan")


def test_distinct_entity_survives():
    result = mod.choose("trivia", "That '70s Show",
                        ["That 70s Show sitcom", "All in the Family"])
    assert result["cluster_key"] == "all in family"
