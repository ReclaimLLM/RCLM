from rclm.compress.filters.lossless import compact_search_and_paths


def test_folds_repeated_search_file_without_dropping_matches() -> None:
    output = "src/app.py:10:first\nsrc/app.py:20:second\nsrc/lib.py:5:third\n"

    assert compact_search_and_paths(output) == (
        "src/app.py\n10:first\n20:second\nsrc/lib.py\n5:third\n"
    )


def test_folds_search_directory_when_each_file_has_one_match() -> None:
    output = "src/api/a.py:10:first\nsrc/api/b.py:20:second\n"

    assert compact_search_and_paths(output) == "src/api/\na.py:10:first\nb.py:20:second\n"


def test_folds_plain_path_listing() -> None:
    output = "src/api/a.py\nsrc/api/b.py\nsrc/web/c.ts\n"

    assert compact_search_and_paths(output) == "src/api/\na.py\nb.py\nsrc/web/\nc.ts\n"


def test_ambiguous_mixed_path_listing_passes_through() -> None:
    output = "src/api/a.py\nnote\nsrc/api/b.py\n"

    assert compact_search_and_paths(output) is None


def test_unknown_and_oversized_output_pass_through() -> None:
    assert compact_search_and_paths("ordinary output\n") is None
    assert compact_search_and_paths("x" * 2_000_001) is None
