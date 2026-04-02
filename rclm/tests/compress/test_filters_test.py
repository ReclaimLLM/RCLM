"""Tests for test runner output filters."""

from rclm.compress.filters.test import filter_test

# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


class TestFilterPytest:
    def test_all_passing(self):
        output = (
            "tests/test_foo.py::test_one PASSED\n"
            "tests/test_foo.py::test_two PASSED\n"
            "tests/test_foo.py::test_three PASSED\n"
            "========================= 3 passed in 0.05s =========================\n"
        )
        result = filter_test("pytest tests/", output)
        assert "3 passed" in result
        # Individual PASSED lines should be stripped
        assert "PASSED" not in result or result.count("\n") < output.count("\n")

    def test_with_failures(self):
        output = (
            "tests/test_foo.py::test_one PASSED\n"
            "tests/test_foo.py::test_two FAILED\n"
            "========================= FAILURES =========================\n"
            "_________________________ test_two _________________________\n"
            "    def test_two():\n"
            ">       assert 1 == 2\n"
            "E       assert 1 == 2\n"
            "========================= short test summary info =========================\n"
            "FAILED tests/test_foo.py::test_two - assert 1 == 2\n"
            "========================= 1 failed, 1 passed in 0.05s =========================\n"
        )
        result = filter_test("pytest tests/", output)
        assert "FAILED" in result
        assert "assert 1 == 2" in result

    def test_python_m_pytest(self):
        output = "========================= 5 passed in 1.23s =========================\n"
        result = filter_test("python -m pytest tests/ -v", output)
        assert "5 passed" in result

    def test_no_match_for_non_test_command(self):
        assert filter_test("git status", "some output") is None


# ---------------------------------------------------------------------------
# npm test / jest
# ---------------------------------------------------------------------------


class TestFilterJsTest:
    def test_all_passing(self):
        output = (
            "PASS  src/app.test.js\n"
            "  ✓ renders correctly (15ms)\n"
            "  ✓ handles click (8ms)\n"
            "\n"
            "Tests:  2 passed, 2 total\n"
        )
        result = filter_test("npm test", output)
        assert "2 passed" in result

    def test_with_failure(self):
        output = (
            "FAIL  src/app.test.js\n"
            "  ● renders correctly\n"
            "    Expected: 'hello'\n"
            "    Received: 'world'\n"
            "\n"
            "Tests:  1 failed, 1 passed, 2 total\n"
        )
        result = filter_test("npm test", output)
        assert "●" in result or "FAIL" in result
        assert "1 failed" in result

    def test_npx_jest(self):
        output = "Tests:  10 passed, 10 total\n"
        result = filter_test("npx jest", output)
        assert "10 passed" in result

    def test_npx_vitest(self):
        output = "Tests:  3 passed, 3 total\n"
        result = filter_test("npx vitest run", output)
        assert "3 passed" in result


# ---------------------------------------------------------------------------
# cargo test
# ---------------------------------------------------------------------------


class TestFilterCargoTest:
    def test_all_passing(self):
        output = (
            "running 3 tests\n"
            "test tests::test_one ... ok\n"
            "test tests::test_two ... ok\n"
            "test tests::test_three ... ok\n"
            "\n"
            "test result: ok. 3 passed; 0 failed; 0 ignored\n"
        )
        result = filter_test("cargo test", output)
        assert "3 passed" in result

    def test_with_failure(self):
        output = (
            "running 2 tests\n"
            "test tests::test_one ... ok\n"
            "test tests::test_two ... FAILED\n"
            "\n"
            "failures:\n"
            "\n"
            "---- tests::test_two stdout ----\n"
            "thread 'tests::test_two' panicked at 'assertion failed'\n"
            "\n"
            "failures:\n"
            "    tests::test_two\n"
            "\n"
            "test result: FAILED. 1 passed; 1 failed; 0 ignored\n"
        )
        result = filter_test("cargo test", output)
        assert "FAILED" in result
        assert "assertion failed" in result
