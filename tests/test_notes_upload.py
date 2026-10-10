import pytest

from notes_upload import MAX_FILE_BYTES, MAX_FILES, remove_uploads, save_uploads, saved_name, validate_uploads

pytestmark = pytest.mark.unit


def test_valid_files_pass():
    assert validate_uploads([("a.md", b"# A"), ("b.TXT", b"plain")]) == []
    assert validate_uploads([]) == []


def test_too_many_files():
    files = [(f"{i}.md", b"x") for i in range(MAX_FILES + 1)]
    assert validate_uploads(files) == [f"Too many files: {MAX_FILES + 1}. Upload at most {MAX_FILES}."]


def test_size_limit_is_inclusive():
    assert validate_uploads([("ok.md", b"x" * MAX_FILE_BYTES)]) == []
    [error] = validate_uploads([("big.md", b"x" * (MAX_FILE_BYTES + 1))])
    assert error.startswith("big.md:") and "limit is 15 KB" in error


@pytest.mark.parametrize("name, data, problem", [
    ("notes.pdf", b"%PDF", "only .md and .txt"),
    ("notes", b"text", "only .md and .txt"),
    ("blank.md", b"  \n\t", "empty"),
    ("latin.md", "café".encode("latin-1"), "not UTF-8"),
])
def test_bad_file(name, data, problem):
    [error] = validate_uploads([(name, data)])
    assert error.startswith(f"{name}:") and problem in error


def test_md_and_txt_with_the_same_stem_clash():
    [error] = validate_uploads([("a.md", b"1"), ("a.txt", b"2")])
    assert "clashes with a.md" in error


def test_saved_name():
    assert saved_name("notes.md") == "notes.md"
    assert saved_name("notes.TXT") == "notes.md"
    assert saved_name("../../etc/evil.md") == "evil.md"
    assert saved_name("C:\\Users\\me\\notes.txt") == "notes.md"


def test_save_then_remove():
    folder = save_uploads([("a.md", b"# A"), ("../b.txt", b"plain")])
    try:
        assert folder.name.startswith("sherpa-notes-")
        assert sorted(p.name for p in folder.iterdir()) == ["a.md", "b.md"]
        assert (folder / "b.md").read_text() == "plain"
    finally:
        remove_uploads(folder)
    assert not folder.exists()


def test_each_save_gets_its_own_folder():
    a, b = save_uploads([("a.md", b"1")]), save_uploads([("a.md", b"2")])
    try:
        assert a != b
    finally:
        remove_uploads(a)
        remove_uploads(b)


@pytest.mark.parametrize("folder", [None, "", "/nonexistent/sherpa-notes-x"])
def test_remove_tolerates_nothing_to_remove(folder):
    remove_uploads(folder)
