from unittest.mock import Mock


def test_health_checks_database_without_calling_kimi(client, app, fake_kimi):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "llm_configured": True}
    assert fake_kimi.calls == []


def test_unhealthy_database_returns_safe_503(client, app, monkeypatch):
    monkeypatch.setattr(
        app.state.database,
        "health",
        Mock(side_effect=RuntimeError("Sensitive database details")),
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "DATABASE_UNHEALTHY",
            "message": "The database health check failed.",
            "details": None,
        }
    }
    assert "Sensitive database details" not in response.text
