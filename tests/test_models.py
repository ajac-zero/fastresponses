from open_responses_server.models import (
    CustomItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    MessageItem,
    ResponsesRequest,
)


def test_string_input_normalizes_to_user_message():
    request = ResponsesRequest.model_validate({"input": "hello"})
    items = request.input_items()
    assert len(items) == 1
    assert isinstance(items[0], MessageItem)
    assert items[0].role == "user"
    assert items[0].text() == "hello"


def test_item_union_parses_known_and_extension_items():
    request = ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "yo"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "acme:telemetry", "id": "t1", "status": "completed"},
            ],
        }
    )
    items = request.input_items()
    assert isinstance(items[0], MessageItem)
    assert items[1].text() == "yo"
    assert isinstance(items[2], FunctionCallItem)
    assert isinstance(items[3], FunctionCallOutputItem)
    assert isinstance(items[4], CustomItem)
    assert items[4].type == "acme:telemetry"


def test_function_tools_filter():
    request = ResponsesRequest.model_validate(
        {
            "input": "x",
            "tools": [
                {"type": "function", "name": "f", "parameters": {"type": "object"}},
                {"type": "acme:web_search"},
            ],
        }
    )
    tools = request.function_tools()
    assert len(tools) == 1
    assert tools[0].name == "f"
