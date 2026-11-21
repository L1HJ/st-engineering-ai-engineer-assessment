import json

import pytest

from agent_harness.tools import DatasetValidationError, ToolRegistry


VALID_DOCUMENT = {"id": "kb-001", "title": "Runbook", "content": "Check the service."}
VALID_SERVICE = {
    "service_name": "checkout",
    "status": "degraded",
    "description": "Elevated errors.",
    "updated_at": "2026-11-21T08:00:00Z",
}


def load_datasets(tmp_path, knowledge, services):
    knowledge_path = tmp_path / "knowledge_base.json"
    services_path = tmp_path / "services.json"
    knowledge_path.write_text(knowledge, encoding="utf-8")
    services_path.write_text(services, encoding="utf-8")
    return ToolRegistry.from_files(knowledge_path, services_path)


def test_duplicate_document_ids_fail_dataset_loading(tmp_path):
    with pytest.raises(DatasetValidationError, match="unique"):
        load_datasets(
            tmp_path,
            json.dumps([VALID_DOCUMENT, VALID_DOCUMENT]),
            json.dumps([VALID_SERVICE]),
        )


def test_duplicate_service_names_fail_dataset_loading(tmp_path):
    with pytest.raises(DatasetValidationError, match="unique"):
        load_datasets(
            tmp_path,
            json.dumps([VALID_DOCUMENT]),
            json.dumps([VALID_SERVICE, VALID_SERVICE]),
        )


def test_invalid_json_fails_instead_of_loading_an_empty_dataset(tmp_path):
    with pytest.raises(DatasetValidationError, match="valid JSON"):
        load_datasets(tmp_path, "[", json.dumps([VALID_SERVICE]))


@pytest.mark.parametrize("status", ["unknown", "offline", "", 42])
def test_invalid_dataset_service_status_fails_loading(tmp_path, status):
    service = {**VALID_SERVICE, "status": status}
    with pytest.raises(DatasetValidationError, match="schema validation"):
        load_datasets(
            tmp_path,
            json.dumps([VALID_DOCUMENT]),
            json.dumps([service]),
        )


@pytest.mark.parametrize(
    ("dataset", "record"),
    [
        ("knowledge", {**VALID_DOCUMENT, "extra": "not allowed"}),
        ("services", {**VALID_SERVICE, "extra": "not allowed"}),
    ])
def test_extra_dataset_fields_fail_loading(tmp_path, dataset, record):
    knowledge = [record] if dataset == "knowledge" else [VALID_DOCUMENT]
    services = [record] if dataset == "services" else [VALID_SERVICE]
    with pytest.raises(DatasetValidationError, match="schema validation"):
        load_datasets(tmp_path, json.dumps(knowledge), json.dumps(services))


@pytest.mark.parametrize(
    "updated_at",
    [
        0,
        "2026-11-21 08:00:00Z",
        "2026-11-21T08:00:00+08:00",
        "2026-11-21T08:00:00-00:00",
        "2026-02-30T08:00:00Z",
        "not-a-time",
        None,
    ],
)
def test_dataset_service_timestamp_must_be_valid_utc_rfc3339(tmp_path, updated_at):
    service = {**VALID_SERVICE, "updated_at": updated_at}
    with pytest.raises(DatasetValidationError, match="schema validation"):
        load_datasets(
            tmp_path,
            json.dumps([VALID_DOCUMENT]),
            json.dumps([service]),
        )


def test_dataset_service_timestamp_accepts_explicit_utc_offset(tmp_path):
    service = {**VALID_SERVICE, "updated_at": "2026-11-21T08:00:00+00:00"}
    registry = load_datasets(
        tmp_path,
        json.dumps([VALID_DOCUMENT]),
        json.dumps([service]),
    )
    assert registry.names == (
        "search_knowledge_base",
        "get_service_status",
        "create_incident",
    )


def test_dataset_top_level_value_must_be_an_array(tmp_path):
    with pytest.raises(DatasetValidationError, match="top-level array"):
        load_datasets(
            tmp_path,
            json.dumps({"documents": [VALID_DOCUMENT]}),
            json.dumps([VALID_SERVICE]),
        )
