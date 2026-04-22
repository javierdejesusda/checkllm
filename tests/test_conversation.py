import pytest

from checkllm.conversation import Turn, ConversationalTestCase


class TestTurn:
    def test_creates_turn(self):
        turn = Turn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_default_metadata(self):
        turn = Turn(role="assistant", content="Hi there")
        assert turn.metadata == {}


class TestConversationalTestCase:
    @pytest.fixture
    def multi_turn_conversation(self):
        return ConversationalTestCase(
            turns=[
                Turn(role="system", content="You are a helpful assistant."),
                Turn(role="user", content="What is Python?"),
                Turn(role="assistant", content="Python is a programming language."),
                Turn(role="user", content="What about Java?"),
                Turn(role="assistant", content="Java is also a programming language."),
            ]
        )

    def test_user_turns(self, multi_turn_conversation):
        user_turns = multi_turn_conversation.user_turns
        assert len(user_turns) == 2
        assert user_turns[0].content == "What is Python?"
        assert user_turns[1].content == "What about Java?"

    def test_assistant_turns(self, multi_turn_conversation):
        assistant_turns = multi_turn_conversation.assistant_turns
        assert len(assistant_turns) == 2
        assert assistant_turns[0].content == "Python is a programming language."
        assert assistant_turns[1].content == "Java is also a programming language."

    def test_last_response(self, multi_turn_conversation):
        assert multi_turn_conversation.last_response == "Java is also a programming language."

    def test_last_response_none_when_no_assistant(self):
        conv = ConversationalTestCase(
            turns=[
                Turn(role="user", content="Hello"),
                Turn(role="user", content="Anyone there?"),
            ]
        )
        assert conv.last_response is None

    def test_format_transcript(self, multi_turn_conversation):
        transcript = multi_turn_conversation.format_transcript()
        assert "[SYSTEM]: You are a helpful assistant." in transcript
        assert "[USER]: What is Python?" in transcript
        assert "[ASSISTANT]: Python is a programming language." in transcript
        assert "[USER]: What about Java?" in transcript
        assert "[ASSISTANT]: Java is also a programming language." in transcript
        # Verify it is newline-separated
        lines = transcript.split("\n")
        assert len(lines) == 5

    def test_turn_count(self, multi_turn_conversation):
        assert multi_turn_conversation.turn_count == 5

    def test_turn_count_empty(self):
        conv = ConversationalTestCase(turns=[])
        assert conv.turn_count == 0

    def test_first_user_message(self, multi_turn_conversation):
        assert multi_turn_conversation.first_user_message == "What is Python?"

    def test_first_user_message_none_when_no_user(self):
        conv = ConversationalTestCase(turns=[Turn(role="system", content="System message")])
        assert conv.first_user_message is None

    def test_system_turns(self, multi_turn_conversation):
        system_turns = multi_turn_conversation.system_turns
        assert len(system_turns) == 1
        assert system_turns[0].content == "You are a helpful assistant."

    def test_turns_by_role(self, multi_turn_conversation):
        tool_turns = multi_turn_conversation.turns_by_role("tool")
        assert len(tool_turns) == 0
        user_turns = multi_turn_conversation.turns_by_role("user")
        assert len(user_turns) == 2

    def test_slice_turns(self, multi_turn_conversation):
        sliced = multi_turn_conversation.slice_turns(1, 3)
        assert len(sliced) == 2
        assert sliced[0].role == "user"
        assert sliced[1].role == "assistant"

    def test_default_metadata_and_expected_outcome(self):
        conv = ConversationalTestCase(turns=[Turn(role="user", content="Hi")])
        assert conv.metadata == {}
        assert conv.expected_outcome is None

    def test_metadata_on_turn(self):
        turn = Turn(role="user", content="Hi", metadata={"source": "test"})
        assert turn.metadata == {"source": "test"}
