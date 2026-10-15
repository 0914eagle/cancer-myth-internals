import json

import pytest

from scripts import resume_frontier_luna as recovery


NOTICE = {'type': 'error', 'message': 'Reconnecting... 2/5 (unexpected status 403 Forbidden: Unknown error, url: wss://chatgpt.com/backend-api/codex/responses)'}


def stream(*extra):
    items = [{'type': 'thread.started', 'thread_id': 'test'}, {'type': 'turn.started'}, NOTICE]
    items.extend(extra)
    return '\n'.join(json.dumps(e) for e in items)


FINAL = {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Answer.'}}
DONE = {'type': 'turn.completed', 'usage': {}}


def test_recovered_connection_keeps_success_and_original_notices():
    parsed = recovery.parse_events(stream(FINAL, DONE), 'Answer.')
    assert NOTICE in parsed


@pytest.mark.parametrize('extra', [[], [FINAL], [FINAL, {'type': 'turn.failed'}],
                                  [{'type': 'item.completed', 'item': {'type': 'command_execution'}}, FINAL, DONE],
                                  [{'type': 'error', 'message': 'Authentication failed'}, FINAL, DONE]])
def test_incomplete_fatal_and_tool_calls_still_rejected(extra):
    with pytest.raises(ValueError):
        recovery.parse_events(stream(*extra), 'Answer.')
