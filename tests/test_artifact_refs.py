from pathlib import Path

from config import GENERATED_DIR, UPLOADS_DIR, artifact_to_public_ref, resolve_artifact_ref


def test_generated_artifact_roundtrip():
    target = (GENERATED_DIR / "run_demo" / "output.py").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('ok')\n", encoding="utf-8")

    ref = artifact_to_public_ref(target)
    assert ref == "/generated/run_demo/output.py"
    assert resolve_artifact_ref(ref) == target


def test_uploaded_artifact_roundtrip():
    target = (UPLOADS_DIR / "sample.pdf").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"%PDF-1.4\n")

    ref = artifact_to_public_ref(target)
    assert ref == "/uploads/sample.pdf"
    assert resolve_artifact_ref(ref) == target
