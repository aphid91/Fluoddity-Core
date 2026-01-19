import numpy as np

# Maximum number of rules to keep in history
MAX_HISTORY_SIZE = 200


class RuleManager:
    """Owns rule_history and all rule operations."""

    def __init__(self):
        self.rule_history: list[np.ndarray] = []

    def push_rule(self, rule: np.ndarray) -> None:
        """Add a new rule to history, trimming oldest if exceeds limit."""
        self.rule_history.append(rule)
        self._trim_history()

    def _trim_history(self) -> None:
        """Remove oldest rules if history exceeds MAX_HISTORY_SIZE."""
        while len(self.rule_history) > MAX_HISTORY_SIZE:
            self.rule_history.pop(0)

    def push_zero_rule(self) -> np.ndarray:
        """Push a zero rule (no target) to history. Returns the zero rule."""
        zero_rule = np.zeros((10, 8), dtype=np.float32)
        self.push_rule(zero_rule)
        return zero_rule

    def pop_rule(self) -> np.ndarray | None:
        """Remove and return the previous rule, or None if empty."""
        if len(self.rule_history) > 1:
            self.rule_history.pop()
            return self.rule_history[-1]
        elif len(self.rule_history) == 1:
            self.rule_history = []
            return None  # Signals to use zero rule
        return None

    def get_current_rule(self) -> np.ndarray | None:
        """Get current rule without modifying history."""
        return self.rule_history[-1] if self.rule_history else None

    def clear(self) -> None:
        """Clear all rule history."""
        self.rule_history = []

    def has_rules(self) -> bool:
        """Check if there are any rules in history."""
        return len(self.rule_history) > 0
