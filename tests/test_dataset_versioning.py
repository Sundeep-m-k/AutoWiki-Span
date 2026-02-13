from pathlib import Path

from fandom_span_id_retrieval.utils.dataset_versioning import build_dataset_manifest


def test_build_dataset_manifest(tmp_path: Path) -> None:
    domain = "demo"
    processed_dir = tmp_path / domain
    processed_dir.mkdir(parents=True)

    (processed_dir / f"pages_{domain}.jsonl").write_text("{}\n", encoding="utf-8")
    (processed_dir / f"paragraphs_{domain}.jsonl").write_text("{}\n", encoding="utf-8")
    (processed_dir / f"paragraph_links_{domain}.csv").write_text("col\n", encoding="utf-8")

    manifest_path = build_dataset_manifest(domain, processed_dir)
    assert manifest_path.exists()

    manifest = manifest_path.read_text(encoding="utf-8")
    assert "dataset_manifest" not in manifest
    assert f"pages_{domain}.jsonl" in manifest
    assert f"paragraphs_{domain}.jsonl" in manifest
