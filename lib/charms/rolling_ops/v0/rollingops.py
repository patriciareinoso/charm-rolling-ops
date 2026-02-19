# Copyright 2026 Canonical Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This library enables "rolling" operations across units of a charmed application using a peer-relation distributed lock.

This library coordinates rolling operations so that at most one unit executes the operation at a time.

For example, a charm author might use this library to implement a "rolling restart", in
which all units in an application restart their workload, but no two units execute the
restart at the same time.

Interface (peer relation):
- Unit databag keys:
  - state: "idle" | "request" | "retry"
  - operations: JSON-encoded list of queued operations (FIFO)
  - last_retry: timestamp string (UTC) of the most recent retry signal
  - attempt: integer (string) number of attempts for the head operation
  - completed_at: timestamp string (UTC) when the head operation completed successfully

- App databag keys:
  - granted_unit: string "<unit-id>" or ""
  - granted_at: timestamp string (UTC) when the lock was granted

Operation queue semantics:
- Units enqueue operations into instead of overwriting a single pending request.
- Deduplication: if the last queued operation has the same callback_key and kwargs,
  the new request is ignored (no-op). Otherwise it is appended.
- Execution fairness: when granted, a unit executes exactly ONE operation (queue head),
  then releases the lock to allow other units to run.

Retry semantics:
- If a unit returns OperationResult.RETRY, it transitions to state="retry", increments attempt,
  and records last_retry. The head operation remains queued.
- A max_retries value may be specified per operation. When exceeded, the head operation is
  dropped and the unit proceeds to attempt to acquire the lock for the next queued operation (if any).

Scheduling:
- The leader grants the lock when no unit is currently granted.
- Requests are preferred over retries.
- Among requests, the oldest enqueued_at is selected.
- Among retries, the oldest last_retry is selected.

All timestamps are stored in UTC using TIMESTAMP_FORMAT.

To implement the rolling restart, a charm author would do the following:

1. Add a peer relation called 'restart' to a charm's `metadata.yaml`:
```yaml
peers:
    restart:
        interface: rolling_op
```

Import this library into src/charm.py, and initialize a RollingOpsManager in the Charm's
`__init__`. The Charm should also define a callback routine, which will be executed when
a unit holds the distributed lock:

src/charm.py
```python
# ...
from charms.rolling_ops.v0.rollingops import RollingOpsManager
# ...
class SomeCharm(...):
    def __init__(...)
        # ...
        self.restart_manager = RollingOpsManager(
            charm=self, relation="restart", callback=self._restart
        )
        # ...
    def _restart(self, event):
        systemd.service_restart('foo')
```

To kick off the rolling restart, emit this library's AcquireLock event. The simplest way
to do so would be with an action, though it might make sense to acquire the lock in
response to another event.

```python
    def _on_trigger_restart(self, event):
        self.charm.on[self.restart_manager.name].acquire_lock.emit()
```

In order to trigger the restart, a human operator would execute the following command on
the CLI:

```
juju run-action some-charm/0 some-charm/1 <... some-charm/n> restart
```

Note that all units that plan to restart must receive the action and emit the acquire
event. Any units that do not run their acquire handler will be left out of the rolling
restart. (An operator might take advantage of this fact to recover from a failed rolling
operation without restarting workloads that were able to successfully restart -- simply
omit the successful units from a subsequent run-action call.)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import AnyStr, Optional

from ops.charm import (
    CharmBase,
    CollectStatusEvent,
    RelationChangedEvent,
    RelationDepartedEvent,
)
from ops.framework import EventBase, Object
from ops.model import ActiveStatus

logger = logging.getLogger(__name__)

# The unique Charmhub library identifier, never change it
LIBID = "20b7777f58fe421e9a223aefc2b4d3a4"

# Increment this major API version when introducing breaking changes
LIBAPI = 0

# Increment this PATCH version before using `charmcraft publish-lib` or reset
# to 0 if you are raising the major API version
LIBPATCH = 8

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%fZ"


def _now_timestamp() -> str:
    """UTC timestamp string with microseconds."""
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def _parse_timestamp(timestamp: str) -> datetime:
    """Parse timestamp string. Return 'now' on errors to avoid selecting invalid timestamps."""
    try:
        return datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    except Exception:
        return datetime.now(timezone.utc)


def _args_to_json(data: dict[str, any]) -> str:
    """Deterministic JSON serialization for kwargs."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


class LockNoRelationError(Exception):
    """Raised if we are trying to process a lock, but do not appear to have a relation yet."""


@dataclass(frozen=True)
class Operation:
    """A single queued operation."""

    callback_id: str
    kwargs: str
    requested_at: str
    max_retry: int

    @classmethod
    def create(
        cls,
        callback_id: str,
        kwargs: dict[str, any],
        max_retry: int = -1,
    ) -> "Operation":
        """Create a new operation from a callback id and kwargs."""
        return cls(
            callback_id=callback_id,
            kwargs=_args_to_json(kwargs),
            requested_at=_now_timestamp(),
            max_retry=max_retry,
        )

    def _to_dict(self) -> dict[str, str]:
        """Dict form (string-only values)."""
        return {
            "callback_id": self.callback_id,
            "kwargs": self.kwargs,
            "requested_at": self.requested_at,
            "max_retry": str(self.max_retry),
        }

    def to_string(self) -> str:
        """Serialize to a string suitable for a Juju databag."""
        return json.dumps(self._to_dict(), separators=(",", ":"))

    def parsed_kwargs(self) -> dict[str, any]:
        """Parsed kwargs for callback execution."""
        return json.loads(self.kwargs) if self.kwargs else {}

    @classmethod
    def from_string(cls, data: str) -> "Operation":
        """Deserialize from a Juju databag string."""
        obj = json.loads(data)
        return cls(
            callback_id=obj["callback_id"],
            kwargs=obj["kwargs"],
            requested_at=obj["requested_at"],
            max_retry=int(obj["max_retry"]),
        )

    def __eq__(self, other: object) -> bool:
        """Equal for the operation."""
        if not isinstance(other, Operation):
            return NotImplemented
        return self.callback_id == other.callback_id and self.kwargs == other.kwargs

    def __hash__(self) -> int:
        """Hash for the operation."""
        return hash((self.callback_id, self.kwargs))


class OperationQueue:
    """In-memory FIFO queue of Operations with encode/decode helpers for storing in a databag."""

    def __init__(self, operations: Optional[list[Operation]] = None):
        self.operations: list[Operation] = list(operations or [])

    # ---- core queue operations ----

    def __len__(self) -> int:
        """Return the number of operations in the queue."""
        return len(self.operations)

    def is_empty(self) -> bool:
        """Return True if there are no queued operations."""
        return not self.operations

    def peek(self) -> Optional[Operation]:
        """Return the first operation in the queue if it exists."""
        return self.operations[0] if self.operations else None

    def _peek_last(self) -> Optional[Operation]:
        """Return the last operation in the queue if it exists."""
        return self.operations[-1] if self.operations else None

    def dequeue(self) -> Optional[Operation]:
        """Drop the first operation in the queue if it exists and return it."""
        return self.operations.pop(0) if self.operations else None

    def _enqueue(self, operation: Operation) -> bool:
        """Append operation only if it is not equal to the last enqueued operation.

        Returns True if added, False if it was already in the queue.
        """
        if last_operation := self._peek_last():
            if last_operation == operation:
                return False
        self.operations.append(operation)
        return True

    def enqueue_lock_request(
        self, callback_id: str, kwargs: dict[str, any], max_retry: int = -1
    ) -> bool:
        """Enqueue a lock request."""
        return self._enqueue(Operation.create(callback_id, kwargs, max_retry=max_retry))

    def to_string(self) -> str:
        """Encode entire queue to a single string."""
        items = [op.to_string() for op in self.operations]
        return json.dumps(items, separators=(",", ":"))

    @classmethod
    def from_string(cls, data: str) -> "OperationQueue":
        """Decode queue from a single string."""
        if not data:
            return cls([])
        items = json.loads(data)
        if not isinstance(items, list):
            raise ValueError("Queue string must decode to a JSON list")
        operations = [Operation.from_string(s) for s in items]
        return cls(operations)


class LockIntent(Enum):
    """Unit-level lock intents stored in unit databags."""

    REQUEST = "request"
    RETRY = "retry"
    IDLE = "idle"


class OperationResult(Enum):
    """Callback return values."""

    COMPLETED = "completed"
    RETRY = "retry"


class Lock:
    """State machine view over peer relation databags for a single unit.

    This class is the only component that should directly read/write the peer relation
    databags for lock state, queue state, and grant state.

    Important:
      - All relation databag values are strings.
      - This class updates both unit databags and app databags, which triggers
        relation-changed events.
    """

    def __init__(self, manager, unit=None):
        self.relation = manager.model.relations[manager.relation_name][0]
        if not self.relation:
            # TODO: defer caller in this case (probably just fired too soon).
            raise LockNoRelationError()

        self.unit = unit or manager.model.unit
        self.app = manager.model.app

    def request(self, callback_id: str, kwargs: dict, max_retry: int | None = -1):
        """Enqueue an operation and mark this unit as requesting the lock.

        Args:
          callback_id: identifies which callback to execute.
          kwargs: dict of callback kwargs.
          max_retry: None -> unlimited retries (-1), else explicit integer.
        """
        queue = OperationQueue.from_string(self.relation.data[self.unit].get("operations", ""))
        if queue.is_empty():
            self.relation.data[self.unit].update({"state": LockIntent.REQUEST.value})
        op_max_retry = -1 if (max_retry is None or max_retry <= -1) else max_retry
        queue.enqueue_lock_request(callback_id, kwargs, max_retry=op_max_retry)
        self.relation.data[self.unit].update({"operations": queue.to_string()})

    def retry(self):
        """Mark retry for the head operation.

        If max_retry is reached, the head operation is dropped via complete().
        """
        if self._is_max_retry_reached():
            logger.info("Operation max retry reached. Dropping")
            self.complete()
            return
        self.relation.data[self.unit].update({
            "last_retry": _now_timestamp(),
            "state": LockIntent.RETRY.value,
        })
        self._increase_attempt()

    def complete(self):
        """Mark the head operation as completed successfully, pop it from the queue.

        Update unit state depending on whether more operations remain.
        """
        queue = OperationQueue.from_string(self.relation.data[self.unit].get("operations", ""))
        queue.dequeue()
        next_state = LockIntent.REQUEST.value if queue.peek() else LockIntent.IDLE.value

        self.relation.data[self.unit].update({
            "state": next_state,
            "last_retry": "",
            "attempt": "",
            "operations": queue.to_string(),
            "completed_at": _now_timestamp(),
        })

    def release(self):
        """Clear the application-level grant."""
        self.relation.data[self.app].update({
            "granted_unit": "",
            "granted_at": "",
        })

    def grant(self) -> None:
        """Grant a lock to a unit."""
        self.relation.data[self.app].update({
            "granted_unit": str(self.unit),
            "granted_at": _now_timestamp(),
        })

    def is_granted(self) -> bool:
        """Return True if the unit holds the lock."""
        granted_unit = self.relation.data[self.app].get("granted_unit", "")
        return granted_unit == str(self.unit)

    def should_run(self) -> bool:
        """Return True if the lock has been granted to the unit and it is time to execute callback."""
        if self.is_held() and self._grant_is_after_unit_completed():  # REQUEST - GRANTED
            return True

        if self.is_retry() and self._grant_is_after_unit_retry():  # RETRY - GRANTED
            return True

    def should_release(self) -> bool:
        """Return True if the unit finished executing the callback and should be released."""
        if self.is_completed():  # IDLE - granted
            return True
        elif self.is_retry() and not self._grant_is_after_unit_retry():  # RETRY - granted
            return True
        elif self.is_held() and not self._grant_is_after_unit_completed():  # REQUEST - granted
            return True
        return False

    def is_held(self) -> bool:
        """Return True if the unit holds the lock."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit = self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.REQUEST.value and granted_unit == str(self.unit)

    def is_waiting(self) -> bool:
        """Return True if this unit is waiting for a lock to be granted."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit = self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.REQUEST.value and granted_unit != str(self.unit)

    def is_completed(self) -> bool:
        """Return True if this unit is completed callback but still has the grant (leader should clear)."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit = self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.IDLE.value and granted_unit == str(self.unit)

    def is_retry(self) -> bool:
        """Return True if this unit requested retry but still has the grant (leader should clear)."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit = self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.RETRY.value and granted_unit == str(self.unit)

    def is_waiting_retry(self) -> bool:
        """Return True if the unit requested retry and is waiting for lock to be granted."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit = self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.RETRY.value and granted_unit != str(self.unit)

    def get_operation(self) -> Operation | None:
        """Return the head operation for this unit, if any."""
        q = OperationQueue.from_string(self.relation.data[self.unit].get("operations", ""))
        return q.peek()

    def _is_max_retry_reached(self) -> bool:
        """Return True if the head operation exceeded its max_retry (unless max_retry < 0)."""
        operation = self.get_operation()
        if not operation:
            return True
        if operation.max_retry < 0:
            return False
        attempt = self.relation.data[self.unit].get("attempt", "")
        attempt_int = int(attempt) if attempt else 0
        return attempt_int >= operation.max_retry

    def _increase_attempt(self) -> None:
        """Increment the attempt counter for the head operation."""
        attempt = self.relation.data[self.unit].get("attempt", "")
        attempt_int = int(attempt) + 1 if attempt else 0
        self.relation.data[self.unit].update({"attempt": str(attempt_int)})

    def get_attempt(self) -> int:
        """Get current attempt counter (0 if absent)."""
        attempt = self.relation.data[self.unit].get("attempt", "")
        return attempt if attempt else 0

    def get_last_retry(self) -> datetime | None:
        """Get the time the unit requested a retry of the head operation."""
        timestamp_str = self.relation.data[self.unit].get("last_retry", "")
        if timestamp_str:
            return _parse_timestamp(timestamp_str)
        return None

    def get_requested_at(self) -> datetime | None:
        """Get the time the head operation was requested at."""
        operation = self.get_operation()
        if not operation:
            return None
        return _parse_timestamp(operation.requested_at)

    def _grant_is_after_unit_retry(self) -> bool:
        if not self.is_retry():
            return True

        grant_ts = _parse_timestamp(self.relation.data[self.app].get("granted_at", ""))
        retry_ts = _parse_timestamp(self.relation.data[self.unit].get("last_retry", ""))

        # If there was never a retry, treat grant as "after"
        if grant_ts is None:
            return False
        if retry_ts is None:
            return True

        return grant_ts > retry_ts

    def _grant_is_after_unit_completed(self) -> bool:
        if not self.is_held():
            return True

        grant_ts = _parse_timestamp(self.relation.data[self.app].get("granted_at", ""))
        completed_ts = _parse_timestamp(self.relation.data[self.unit].get("completed_at", ""))

        if grant_ts is None:
            return False
        if completed_ts is None:
            return True

        return grant_ts > completed_ts


class Locks:
    """Iterator over Lock objects for each unit present on the peer relation."""

    def __init__(self, manager):
        relation = manager.model.relations[manager.relation_name][0]
        units = list(relation.units)
        units.append(manager.model.unit)
        self._units = units
        self._manager = manager

    def __iter__(self):
        """Yields a lock for each unit we can find on the relation."""
        for unit in self._units:
            yield Lock(self._manager, unit=unit)


class AcquireLock(EventBase):
    """Signals that this unit wants to acquire a lock."""

    def __init__(
        self, handle, callback_id: str, kwargs: dict[str, any] = {}, max_retry: int | None = None
    ):
        super().__init__(handle)
        self.callback_id = callback_id
        self.kwargs = kwargs
        self.max_retry = max_retry

    def snapshot(self):
        """Snapshot of lock event."""
        return {
            "callback_id": self.callback_id,
            "kwargs": self.kwargs,
            "max_retry": self.max_retry,
        }

    def restore(self, snapshot):
        """Restores lock event."""
        self.callback_id = snapshot["callback_id"]
        self.kwargs = snapshot["kwargs"]
        self.max_retry = snapshot["max_retry"]


def pick_oldest_retry(locks: list[Lock]) -> Optional[Lock]:
    """Choose the retry lock with the oldest last_retry timestamp."""
    selected = None
    oldest_timestamp = None

    for lock in locks:
        timestamp = lock.get_last_retry()
        if not timestamp:
            continue

        if oldest_timestamp is None or timestamp < oldest_timestamp:
            oldest_timestamp = timestamp
            selected = lock

    return selected


def pick_oldest_request(locks: list["Lock"]) -> Optional["Lock"]:
    """Choose the lock with the oldest head operation."""
    selected = None
    oldest_request = None

    for lock in locks:
        timestamp = lock.get_requested_at()
        if not timestamp:
            continue

        if oldest_request is None or timestamp < oldest_request:
            oldest_request = timestamp
            selected = lock

    return selected


class ProcessLocks(EventBase):
    """Used to tell the leader to process all locks."""


class RollingOpsManager(Object):
    """Emitters and handlers for rolling ops."""

    def __init__(self, charm: CharmBase, relation: AnyStr, callback_targets=dict[str, any]):
        """Register our custom events.

        params:
            charm: the charm we are attaching this to.
            relation: the peer relation name from metadata.yaml.
            callback: mapping from callback_id -> callable.
        """
        super().__init__(charm, None)
        self.charm = charm
        self.relation_name = relation
        self.callback_targets = callback_targets

        self.framework.observe(
            charm.on[self.relation_name].relation_changed, self._on_relation_changed
        )
        self.framework.observe(charm.on.leader_elected, self._process_locks)
        # self.framework.observe(charm.on.update_status, self._trigger_leader_retry)
        self.framework.observe(
            charm.on[self.relation_name].relation_departed, self._on_relation_departed
        )

    def _on_relation_departed(self, event: RelationDepartedEvent):
        """Leader cleanup: if a departing unit was granted, clear the grant.

        This prevents deadlocks when the granted unit leaves the relation.
        """
        if not self.model.unit.is_leader():
            return
        if unit := event.departing_unit:
            lock = Lock(self, unit)
            if lock.is_granted():
                lock.release()
                self._process_locks()

    def _trigger_leader_retry(self, event: CollectStatusEvent):
        if not self.model.unit.is_leader():
            return

        relation = self.model.get_relation(self.relation_name)
        if not relation:
            return

        lock = Lock(self)
        if lock.should_run():
            logger.info("Running operation on leader unit.")
            self._run_with_lock()

    def _on_relation_changed(self: CharmBase, event: RelationChangedEvent):
        """Process relation changed."""
        lock = Lock(self)

        if lock.should_run():
            self._run_with_lock()
            return

        if self.model.unit.is_leader():
            self._process_locks()

    def _process_locks(self, _: EventBase = None):
        """Process locks."""
        if not self.model.unit.is_leader():
            return

        for lock in Locks(self):
            if lock.should_release():
                lock.release()
            break

        relation = self.model.get_relation(self.relation_name)
        granted_unit = relation.data[self.model.app].get("granted_unit", "")

        if granted_unit:
            logger.info("Current granted_unit=%s. No new unit will be scheduled.", granted_unit)
            return  # REQUEST - GRANTED -> running get out

        self._schedule()

    def _schedule(self) -> None:
        logger.info("Starting scheduling")

        pending_requests = []
        pending_retries = []

        for lock in Locks(self):
            unit_state = lock.relation.data[lock.unit].get("state", "")
            granted_unit = lock.relation.data[lock.app].get("granted_unit", "")

            logger.info(f"PROCESSING {lock.unit} unit={unit_state} app={granted_unit}")

            if lock.is_waiting():  # REQUEST - none
                pending_requests.append(lock)

            elif lock.is_waiting_retry():  # RETRY - none
                pending_retries.append(lock)

        logger.info(f"pending_requests {pending_requests}")
        logger.info(f"pending_retries {pending_retries}")

        selected = None
        if pending_requests:
            selected = pick_oldest_request(pending_requests)
        elif pending_retries:
            selected = pick_oldest_retry(pending_retries)

        if not selected:
            self.model.app.status = ActiveStatus()
            return

        selected.grant()
        if selected.unit == self.model.unit and not selected.is_retry():
            self._run_with_lock()  # REQUEST - granted
            return

    def request_lock(
        self: CharmBase,
        callback_id: str,
        kwargs: dict[str, any] = {},
        max_retry: int | None = None,
    ):
        """Request a lock."""
        try:
            lock = Lock(self)
            lock.request(callback_id, kwargs, max_retry)

            if self.model.unit.is_leader():
                relation = self.model.get_relation(self.relation_name)
                self.charm.on[self.relation_name].relation_changed.emit(
                    relation, app=self.charm.app
                )

        except LockNoRelationError:
            logger.debug(
                "No {} peer relation yet. Delaying rolling op.".format(self.relation_name)
            )

    def _run_with_lock(self: CharmBase):
        lock = Lock(self)

        operation = lock.get_operation()
        if not operation:
            lock.complete()
            if self.model.unit.is_leader():
                self._process_locks()
                return

        callback = self.callback_targets.get(operation.callback_id)
        kwargs = operation.parsed_kwargs()

        logger.info(
            "Executing attempt=%s callback_id=%s", lock.get_attempt(), operation.callback_id
        )

        try:
            result = callback(**kwargs)
        except Exception as e:
            logger.error("Operation failed: %s: %s", operation.callback_id, e)
            result = OperationResult.RETRY

        if result == OperationResult.RETRY:
            lock.retry()
            if self.model.unit.is_leader():
                self._process_locks()
            return

        lock.complete()
        if self.model.unit.is_leader():
            self._process_locks()
