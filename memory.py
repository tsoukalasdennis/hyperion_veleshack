class ConversationMemory:
    """Simple in-memory conversation history per user/session."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.sessions: dict[str, list[dict[str, str]]] = {}

    def add_user_message(
        self,
        user_id: str,
        message: str,
    ) -> None:
        history = self.sessions.setdefault(user_id, [])

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        self._trim(user_id)

    def add_assistant_message(
        self,
        user_id: str,
        message: str,
    ) -> None:
        history = self.sessions.setdefault(user_id, [])

        history.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

        self._trim(user_id)

    def get_history(
        self,
        user_id: str,
    ) -> list[dict[str, str]]:
        return self.sessions.get(user_id, []).copy()

    def format_history(
        self,
        user_id: str,
    ) -> str:
        history = self.get_history(user_id)

        if not history:
            return ""

        return "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in history
        )

    def _trim(
        self,
        user_id: str,
    ) -> None:
        history = self.sessions[user_id]

        max_messages = self.max_turns * 2

        if len(history) > max_messages:
            self.sessions[user_id] = history[-max_messages:]
