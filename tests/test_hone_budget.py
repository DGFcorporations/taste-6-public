"""Contract bound: hone stops after 15 calls for one task."""


def test_hone_budget_is_fifteen_calls_per_task(client):
    body = {
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "task_ref": "budget-1"},
        "draft": "Lead with the verdict.",
    }
    for index in range(15):
        segment = {**body["segment_ctx"], "segment": f"part-{index % 3}"}
        response = client.post("/hone", json={**body, "segment_ctx": segment})
        assert response.status_code == 200, response.text
    blocked = client.post("/hone", json=body)
    assert blocked.status_code == 429
    error = blocked.json()["error"]
    assert error["code"] == "e429"
    assert "budget" in error["message"]


def test_validation_message_is_plain_text(client):
    response = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {}, "pick": "A", "rationale": "   ",
    })
    assert response.status_code == 422
    message = response.json()["error"]["message"]
    assert not message.startswith("[")
    assert "rationale" in message
