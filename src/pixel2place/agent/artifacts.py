"""새 실행 디렉터리의 산출물 등록과 변조 검사."""

import hashlib
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from pixel2place.agent.contracts import Contract, OpaqueId

type ArtifactKind = Literal["inputs", "layout", "calibration", "check", "image_check"]


class ArtifactError(ValueError):
    """산출물이 현재 실행에 속하지 않거나 파일이 바뀌었다."""


class ArtifactFile(Contract):
    """실행 디렉터리 기준 상대 경로와 파일 해시."""

    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ArtifactRecord(Contract):
    """첫 파일은 주 산출물이다. 나머지는 함께 검증하는 근거다."""

    id: OpaqueId
    kind: ArtifactKind
    layout_id: OpaqueId | None
    files: tuple[ArtifactFile, ...] = Field(min_length=1)


class Manifest(Contract):
    """현재 실행에서 등록한 산출물만 보존한다."""

    schema_version: Literal[1] = 1
    run_id: OpaqueId
    artifacts: tuple[ArtifactRecord, ...]


def write_record(path: Path, model: BaseModel) -> None:
    """같은 디렉터리의 임시 파일을 완성한 뒤 목적 파일로 교체한다."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(model.model_dump_json(indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactStore:
    """기존 디렉터리를 열지 않는다. ID는 메모리의 등록 목록에서만 찾는다."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=False)
        self.root = root.resolve()
        self.run_id = uuid4().hex
        self._records: dict[str, ArtifactRecord] = {}
        self._manifest_text = ""
        self._save_manifest()

    def register(
        self, kind: ArtifactKind, files: tuple[Path, ...], *, layout_id: str | None = None
    ) -> ArtifactRecord:
        """모든 파일을 검사하고 새 ID를 등록한다. 실패하면 ID를 반환하지 않는다."""
        self._verify_manifest()
        record = ArtifactRecord(
            id=uuid4().hex,
            kind=kind,
            layout_id=layout_id,
            files=tuple(self._describe(path) for path in files),
        )
        self._records[record.id] = record
        self._save_manifest()
        return record

    def require(
        self, artifact_id: str, kind: ArtifactKind, *, layout_id: str | None = None
    ) -> Path:
        """종류, 소속과 모든 해시를 검증한 뒤 주 파일 경로를 반환한다."""
        self._verify_manifest()
        record = self._records.get(artifact_id)
        if record is None or record.kind != kind or record.layout_id != layout_id:
            raise ArtifactError("The output file does not belong to this tool call.")
        for file in record.files:
            if self._describe(self.root / file.path) != file:
                raise ArtifactError("The output file has changed.")
        return self.root / record.files[0].path

    def _describe(self, path: Path) -> ArtifactFile:
        """링크, 디렉터리 밖의 파일과 없는 파일을 거부한다."""
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError("The output path is outside the run directory.") from exc
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ArtifactError("The output path contains a symbolic link.")
            if component == self.root:
                break
        if not path.is_file() or path.resolve() != path:
            raise ArtifactError("The output file is unavailable.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return ArtifactFile(path=relative.as_posix(), sha256=digest)

    def _save_manifest(self) -> None:
        manifest = Manifest(run_id=self.run_id, artifacts=tuple(self._records.values()))
        write_record(self.root / "manifest.json", manifest)
        self._manifest_text = manifest.model_dump_json(indent=2) + "\n"

    def _verify_manifest(self) -> None:
        path = self.root / "manifest.json"
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_text(encoding="utf-8") != self._manifest_text
        ):
            raise ArtifactError("The output records (manifest.json) have changed.")
