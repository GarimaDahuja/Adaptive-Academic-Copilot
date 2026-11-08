def test_health_check_returns_200(client):
    """Verify that the /api/health endpoint returns HTTP 200 OK."""
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_check_payload(client):
    """Verify that the /api/health endpoint returns expected JSON payload."""
    response = client.get("/api/health")
    data = response.get_json()

    assert data is not None
    assert data.get("status") == "healthy"
    assert "service" in data
    assert "version" in data


def test_404_not_found_returns_json(client):
    """Verify that requesting an unregistered route returns a clean 404 JSON response."""
    response = client.get("/api/nonexistent-endpoint")
    assert response.status_code == 404

    data = response.get_json()
    assert data is not None
    assert data.get("error") == "Not Found"

