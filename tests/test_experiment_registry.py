from pathlib import Path

from fandom_span_id_retrieval.utils.experiment_registry import (
    build_run_entry,
    create_run_id,
    init_run_dir,
    write_registry_entry,
    write_run_metadata,
)


def test_registry_helpers(tmp_path: Path) -> None:
    run_id = create_run_id("unit")
    assert run_id.startswith("unit_")

    run_info = init_run_dir(tmp_path, run_id)
    assert run_info.run_dir.exists()

    meta_path = write_run_metadata(run_info.run_dir, {"ok": True})
    assert meta_path.exists()

    entry = build_run_entry(tmp_path, run_info, "test", [])
    reg_path = write_registry_entry(tmp_path, entry)
    assert reg_path.exists()
