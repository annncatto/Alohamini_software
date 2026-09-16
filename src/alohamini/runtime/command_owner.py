# SPDX-License-Identifier: Apache-2.0
# Adapted from the AlohaMini Host command_owner.py and ingress identity validation.
# Changes: typed identity, bound Host session, no anonymous legacy command fallback.
"""Single-writer arbitration; the Host must stop motion before releasing ownership."""

from collections import OrderedDict

from alohamini._validation import identifier
from alohamini.schema import CommandIdentity


class CommandOwner:
    """Host-thread-only ownership and replay checks, not a watchdog or motor controller.

    Call accept() only after validating the command targets. Each Host restart must
    use a new session ID. release() may only follow a successful stop operation;
    this class does not stop hardware or verify that motion has stopped.
    """

    def __init__(self, host_session_id: str) -> None:
        identifier(host_session_id, "host_session_id")
        self._host_session_id = host_session_id
        self._owner: str | None = None
        self._epoch = 0
        self._sequences: OrderedDict[str, int] = OrderedDict()

    @property
    def host_session_id(self) -> str:
        return self._host_session_id

    @property
    def owner(self) -> str | None:
        return self._owner

    @property
    def epoch(self) -> int:
        return self._epoch

    def accept(self, command: CommandIdentity) -> bool:
        if not isinstance(command, CommandIdentity):
            raise TypeError("Expected a validated CommandIdentity")
        if command.host_session_id != self._host_session_id or command.control_epoch != self._epoch:
            return False
        if self._owner is not None and self._owner != command.client_id:
            return False
        if command.sequence <= self._sequences.get(command.client_id, -1):
            return False
        self._sequences[command.client_id] = command.sequence
        self._sequences.move_to_end(command.client_id)
        if len(self._sequences) > 256:
            self._sequences.popitem(last=False)
        self._owner = command.client_id
        return True

    def release(self) -> None:
        self._owner = None
        self._epoch += 1
