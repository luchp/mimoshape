import types
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import make_figures
from scripts.publish_release import (
    ReleaseAbort,
    ReleaseState,
    describe_publish_actions,
    determine_publish_actions,
    load_code_metadata,
)


def test_make_figures_dispatches_to_selected_paper(monkeypatch):
    called = {"ok": False}

    def fake_main():
        called["ok"] = True

    fake_module = types.SimpleNamespace(main=fake_main)
    monkeypatch.setattr(
        make_figures.importlib, "import_module", lambda _: fake_module
    )

    dispatch = make_figures.load_dispatch_target("26293")
    dispatch()
    assert called["ok"] is True


def test_make_figures_rejects_invalid_paper_id():
    with pytest.raises(SystemExit, match="Invalid paper id"):
        make_figures.load_dispatch_target("26293/../oops")


def test_load_code_metadata_accepts_list_keywords(tmp_path: Path):
    metadata_file = tmp_path / "code.json"
    metadata_file.write_text(
        (
            "{"
            '"title":"t","summary":"s","repository":"o/r",'
            '"author_given_names":"a","author_family_names":"b","affiliation":"c",'
            '"keywords":["k1","k2"],"license":"MIT","message":"m","version":"1.0"'
            "}"
        ),
        encoding="utf-8",
    )
    loaded = load_code_metadata(metadata_file, code_version_override=None)
    assert loaded.version == "1.0"
    assert loaded.keywords == ["k1", "k2"]


def test_determine_publish_actions_idempotent_when_release_exists():
    state = ReleaseState(
        local_code_tag_commit="abc",
        remote_code_tag_commit="abc",
        local_paper_tag_commit="abc",
        remote_paper_tag_commit="abc",
        paper_release_exists=True,
    )
    actions = determine_publish_actions(
        head_commit="abc",
        state=state,
        execute_publish=True,
    )
    assert actions.create_code_tag_local is False
    assert actions.push_code_tag is False
    assert actions.create_paper_tag_local is False
    assert actions.push_paper_tag is False
    assert actions.create_paper_release is False
    assert actions.upload_assets is False


def test_determine_publish_actions_aborts_if_existing_tag_points_elsewhere():
    state = ReleaseState(
        local_code_tag_commit="old",
        remote_code_tag_commit="old",
        local_paper_tag_commit=None,
        remote_paper_tag_commit=None,
        paper_release_exists=False,
    )
    with pytest.raises(ReleaseAbort, match="Refusing to overwrite existing release identity"):
        determine_publish_actions(
            head_commit="new",
            state=state,
            execute_publish=False,
        )


def test_determine_publish_actions_aborts_if_release_exists_without_code_tag():
    state = ReleaseState(
        local_code_tag_commit=None,
        remote_code_tag_commit=None,
        local_paper_tag_commit="abc",
        remote_paper_tag_commit="abc",
        paper_release_exists=True,
    )
    with pytest.raises(ReleaseAbort, match="code tag does not exist"):
        determine_publish_actions(
            head_commit="abc",
            state=state,
            execute_publish=True,
        )


def test_describe_publish_actions_lists_planned_steps():
    state = ReleaseState(
        local_code_tag_commit=None,
        remote_code_tag_commit=None,
        local_paper_tag_commit=None,
        remote_paper_tag_commit=None,
        paper_release_exists=False,
    )
    actions = determine_publish_actions(
        head_commit="abc",
        state=state,
        execute_publish=True,
    )
    plan = describe_publish_actions(actions, code_tag="v1.2.3", paper_tag="paper-26293-v1.0")
    assert plan == [
        "create local annotated tag v1.2.3",
        "push tag v1.2.3 to origin",
        "create local annotated tag paper-26293-v1.0",
        "push tag paper-26293-v1.0 to origin",
        "create GitHub release for paper-26293-v1.0",
        "upload paper PDF and PROVENANCE.txt as release assets",
    ]


def test_describe_publish_actions_reports_when_nothing_needed():
    state = ReleaseState(
        local_code_tag_commit="abc",
        remote_code_tag_commit="abc",
        local_paper_tag_commit="abc",
        remote_paper_tag_commit="abc",
        paper_release_exists=True,
    )
    actions = determine_publish_actions(
        head_commit="abc",
        state=state,
        execute_publish=True,
    )
    plan = describe_publish_actions(actions, code_tag="v1.2.3", paper_tag="paper-26293-v1.0")
    assert plan == ["no publish actions are required; tags and release already exist"]
