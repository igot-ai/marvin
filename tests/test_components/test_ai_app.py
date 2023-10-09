import jsonpatch
import pytest
from marvin.components.ai_application import (
    AIApplication,
    AppPlan,
    FreeformState,
    TaskState,
    UpdatePlan,
    UpdateState,
)
from marvin.tools import Tool
from marvin.utilities.messages import Message

from tests.utils.mark import pytest_mark_class


class GetSchleeb(Tool):
    name = "get_schleeb"

    async def run(self):
        """Get the value of schleeb"""
        return 42


class TestStateJSONPatch:
    def test_update_app_state_valid_patch(self):
        app = AIApplication(
            state=FreeformState(state={"foo": "bar"}), description="test app"
        )
        tool = UpdateState(app=app)
        tool.run([{"op": "replace", "path": "/state/foo", "value": "baz"}])
        assert app.state.dict() == {"state": {"foo": "baz"}}

    def test_update_app_state_invalid_patch(self):
        app = AIApplication(
            state=FreeformState(state={"foo": "bar"}), description="test app"
        )
        tool = UpdateState(app=app)
        with pytest.raises(jsonpatch.InvalidJsonPatch):
            tool.run([{"op": "invalid_op", "path": "/state/foo", "value": "baz"}])
        assert app.state.dict() == {"state": {"foo": "bar"}}

    def test_update_app_state_non_existent_path(self):
        app = AIApplication(
            state=FreeformState(state={"foo": "bar"}), description="test app"
        )
        tool = UpdateState(app=app)
        with pytest.raises(jsonpatch.JsonPatchConflict):
            tool.run([{"op": "replace", "path": "/state/baz", "value": "qux"}])
        assert app.state.dict() == {"state": {"foo": "bar"}}


@pytest_mark_class("llm")
class TestUpdateState:
    def test_keep_app_state(self):
        app = AIApplication(
            name="location tracker app",
            state=FreeformState(state={"San Francisco": {"visited": False}}),
            description="keep track of where I've visited",
        )

        app("I just visited to San Francisco")
        assert bool(app.state.state.get("San Francisco", {}).get("visited"))

        app("oh also I visited San Jose!")

        assert bool(app.state.state.get("San Jose", {}).get("visited"))

    def test_keep_app_state_undo_previous_patch(self):
        app = AIApplication(
            name="location tracker app",
            state=FreeformState(state={"San Francisco": {"visited": False}}),
            description="keep track of where I've visited",
        )

        app("I just visited San Francisco")
        assert bool(app.state.state.get("San Francisco", {}).get("visited"))

        app(
            "sorry, I was confused, I didn't visit San Francisco - but I did visit San"
            " Jose"
        )

        assert not bool(app.state.state.get("San Francisco", {}).get("visited"))
        assert bool(app.state.state.get("San Jose", {}).get("visited"))


class TestPlanJSONPatch:
    def test_update_app_plan_valid_patch(self):
        app = AIApplication(
            plan=AppPlan(
                tasks=[{"id": 1, "description": "test task", "state": "IN_PROGRESS"}]
            ),
            description="test app",
        )
        tool = UpdatePlan(app=app)
        tool.run([{"op": "replace", "path": "/tasks/0/state", "value": "COMPLETED"}])
        assert app.plan.dict() == {
            "tasks": [
                {
                    "id": 1,
                    "description": "test task",
                    "state": TaskState.COMPLETED,
                    "upstream_task_ids": None,
                    "parent_task_id": None,
                }
            ],
            "notes": [],
        }

    def test_update_app_plan_invalid_patch(self):
        app = AIApplication(
            plan=AppPlan(
                tasks=[{"id": 1, "description": "test task", "state": "IN_PROGRESS"}]
            ),
            description="test app",
        )
        tool = UpdatePlan(app=app)
        with pytest.raises(jsonpatch.JsonPatchException):
            tool.run(
                [{"op": "invalid_op", "path": "/tasks/0/state", "value": "COMPLETED"}]
            )
        assert app.plan.dict() == {
            "tasks": [
                {
                    "id": 1,
                    "description": "test task",
                    "state": TaskState.IN_PROGRESS,
                    "upstream_task_ids": None,
                    "parent_task_id": None,
                }
            ],
            "notes": [],
        }

    def test_update_app_plan_non_existent_path(self):
        app = AIApplication(
            plan=AppPlan(
                tasks=[{"id": 1, "description": "test task", "state": "IN_PROGRESS"}]
            ),
            description="test app",
        )
        tool = UpdatePlan(app=app)
        with pytest.raises(jsonpatch.JsonPointerException):
            tool.run(
                [{"op": "replace", "path": "/tasks/1/state", "value": "COMPLETED"}]
            )
        assert app.plan.dict() == {
            "tasks": [
                {
                    "id": 1,
                    "description": "test task",
                    "state": TaskState.IN_PROGRESS,
                    "upstream_task_ids": None,
                    "parent_task_id": None,
                }
            ],
            "notes": [],
        }


@pytest_mark_class("llm")
class TestUpdatePlan:
    def test_keep_app_plan(self):
        app = AIApplication(
            name="Zoo planner app",
            plan=AppPlan(
                tasks=[
                    {
                        "id": 1,
                        "description": "Visit tigers",
                        "state": TaskState.IN_PROGRESS,
                    },
                    {
                        "id": 2,
                        "description": "Visit giraffes",
                        "state": TaskState.PENDING,
                    },
                ]
            ),
            description="plan and track my visit to the zoo",
        )

        app(
            "Actually I heard the tigers ate Carol Baskin's husband - I think I'll skip"
            " visiting them."
        )

        assert [task["state"] for task in app.plan.dict()["tasks"]] == [
            TaskState.SKIPPED,
            TaskState.PENDING,
        ]

        app("Dude i just visited the giraffes!")

        assert [task["state"] for task in app.plan.dict()["tasks"]] == [
            TaskState.SKIPPED,
            TaskState.COMPLETED,
        ]


@pytest_mark_class("llm")
class TestUseCallable:
    def test_use_sync_fn(self):
        def get_schleeb():
            return 42

        app = AIApplication(
            name="Schleeb app",
            tools=[get_schleeb],
            state_enabled=False,
            plan_enabled=False,
            description="answer user questions",
        )

        assert "42" in app("what is the value of schleeb?").content

    def test_use_async_fn(self):
        async def get_schleeb():
            return 42

        app = AIApplication(
            name="Schleeb app",
            tools=[get_schleeb],
            state_enabled=False,
            plan_enabled=False,
            description="answer user questions",
        )

        assert "42" in app("what is the value of schleeb?").content


@pytest_mark_class("llm")
class TestUseTool:
    def test_use_tool(self):
        app = AIApplication(
            name="Schleeb app",
            tools=[GetSchleeb()],
            state_enabled=False,
            plan_enabled=False,
            description="answer user questions",
        )

        assert "42" in app("what is the value of schleeb?").content


@pytest_mark_class("llm")
class TestStreaming:
    def test_streaming(self):
        external_state = {"content": []}

        def handler(chunk):
            delta = chunk["choices"][0]["delta"]
            if delta and delta["content"]:
                external_state["content"].append(delta["content"])

        app = AIApplication(
            name="streaming app",
            stream_handler=handler,
            state_enabled=False,
            plan_enabled=False,
        )

        response = app(
            "say the words 'Hello world' EXACTLY as i have written them."
            " no other characters should be included, do not add any punctuation."
        )

        assert isinstance(response, Message)
        assert "Hello world" in response.content

        assert len([i for i in external_state["content"] if i is not None]) == 2
