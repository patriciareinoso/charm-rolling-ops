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

"""This library enables "rolling" operations across units of a charmed Application.

For example, a charm author might use this library to implement a "rolling restart", in
which all units in an application restart their workload, but no two units execute the
restart at the same time.

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

import logging
from enum import Enum
import json
import subprocess
from typing import AnyStr, Callable, Optional
from ops.framework import EventBase, EventSource, Object
from ops.charm import ActionEvent, CharmBase, RelationChangedEvent, RelationDepartedEvent
from ops.framework import EventBase, Object
from ops.model import ActiveStatus, MaintenanceStatus, WaitingStatus
from ops.charm import CharmBase, CharmEvents
from datetime import datetime, timezone
from dataclasses import dataclass
logger = logging.getLogger(__name__)
import os

# The unique Charmhub library identifier, never change it
LIBID = "20b7777f58fe421e9a223aefc2b4d3a4"

# Increment this major API version when introducing breaking changes
LIBAPI = 0

# Increment this PATCH version before using `charmcraft publish-lib` or reset
# to 0 if you are raising the major API version
LIBPATCH = 8

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%fZ"

class LockNoRelationError(Exception):
    """Raised if we are trying to process a lock, but do not appear to have a relation yet."""

    pass


class DeferLock(Exception):
    """Raised if we are trying to process a lock, but do not appear to have a relation yet."""

    pass


def now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)

def args_to_json(data: dict[str, any]) -> str:
    """
    Deterministic JSON serialization for kwargs.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Operation:
    """
    A single queued operation.
    """
    callback_id: str
    kwargs: str
    request_time: str
    max_retry: int

    @classmethod
    def create(
        cls,
        callback_id: str,
        kwargs: dict[str, any],
        max_retry: int = 0,
    ) -> "Operation":
        """
        Create a new operation from a callback id and kwargs.
        """
        return cls(
            callback_id=callback_id,
            kwargs=args_to_json(kwargs),
            request_time=now_timestamp(),
            max_retry=max_retry,
        )

    def to_dict(self) -> dict[str, str]:
        """
        Dict form (string-only values, Juju-safe).
        """
        return {
            "callback_id": self.callback_id,
            "kwargs": self.kwargs,
            "request_time": self.request_time,
            "max_retry": str(self.max_retry),
        }

    def to_string(self) -> str:
        """
        Serialize to a string suitable for a Juju databag.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_string(cls, data: str) -> "Operation":
        """
        Deserialize from a Juju databag string.
        """
        obj = json.loads(data)
        return cls(
            callback_id=obj["callback_id"],
            kwargs=obj["kwargs"],
            request_time=obj["request_time"],
            max_retry=int(obj["max_retry"]),
        )

    def parsed_kwargs(self) -> dict[str, any]:
        """
        Parsed kwargs for callback execution.
        """
        return json.loads(self.kwargs)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operation):
            return NotImplemented
        return (
            self.callback_id == other.callback_id
            and self.kwargs == other.kwargs
        )

    def __hash__(self) -> int:
        return hash((self.callback_id, self.kwargs))
    
class OperationQueue:
    """
    In-memory FIFO queue of Operations with
    encode/decode helpers for storing in a databag.
    """

    def __init__(self, operations: Optional[list[Operation]] = None):
        self.operations: list[Operation] = list(operations or [])

    # ---- core queue operations ----

    def __len__(self) -> int:
        return len(self.operations)

    def is_empty(self) -> bool:
        return not self.operations

    def peek(self) -> Optional[Operation]:
        return self.operations[0] if self.operations else None
    
    def peek_last(self) -> Optional[Operation]:
        return self.operations[-1] if self.operations else None

    def dequeue(self) -> Optional[Operation]:
        return self.operations.pop(0) if self.operations else None

    def list(self) -> list[Operation]:
        return list(self.operations)

    def enqueue(self, operation: Operation) -> bool:
        """
        Append operation only if it is not equal to the last operation
        Returns True if added, False if it was already in the queue.
        """
        if last_operation:= self.peek_last():
            if last_operation == operation:
                return False
        self.operations.append(operation)
        return True

    def enqueue_request(self, callback_id: str, kwargs: dict[str, any], max_retry: int = -1) -> bool:
        return self.enqueue(Operation.create(callback_id, kwargs, max_retry=max_retry))

    def to_string(self) -> str:
        """
        Encode entire queue to a single string.
        Safe to store in a Juju databag value.
        """
        items = [op.to_string() for op in self.operations]
        return json.dumps(items, separators=(",", ":"))

    @classmethod
    def from_string(cls, data: str) -> "OperationQueue":
        """
        Decode queue from a single string.
        """
        if not data:
            return cls([])
        items = json.loads(data)
        if not isinstance(items, list):
            raise ValueError("Queue string must decode to a JSON list")
        operations = [Operation.from_string(s) for s in items]
        return cls(operations)

class LockIntent(Enum):
    """Possible states for our Distributed lock.

    Note that there are two states set on the unit, and two on the application.

    """
    REQUEST = "request"
    RETRY = "retry"
    IDLE = "idle"

class OperationResult(Enum):
    COMPLETED = "completed"
    RETRY = "retry"

class Lock:
    """A class that keeps track of a single asynchronous lock.

    Warning: a Lock has permission to update relation data, which means that there are
    side effects to invoking the .acquire, .release and .grant methods. Running any one of
    them will trigger a RelationChanged event, once per transition from one internal
    status to another.

    This class tracks state across the cloud by implementing a peer relation
    interface. There are two parts to the interface:

    1) The data on a unit's peer relation (defined in metadata.yaml.) Each unit can update
       this data. The only meaningful values are "acquire", and "release", which represent
       a request to acquire the lock, and a request to release the lock, respectively.

    2) The application data in the relation. This tracks whether the lock has been
       "granted", Or has been released (and reverted to idle). There are two valid states:
       "granted" or None.  If a lock is in the "granted" state, a unit should emit a
       RunWithLocks event and then release the lock.

       If a lock is in "None", this means that a unit has not yet requested the lock, or
       that the request has been completed.

    In more detail, here is the relation structure:

    relation.data:
        <unit n>:
            status: 'acquire|release'
        <application>:
           <unit n>: 'granted|None'

    Note that this class makes no attempts to timestamp the locks and thus handle multiple
    requests in a row. If a unit re-requests a lock before being granted the lock, the
    lock will simply stay in the "acquire" state. If a unit wishes to clear its lock, it
    simply needs to call lock.release().

    """

    def __init__(self, manager, unit=None):
        self.relation = manager.model.relations[manager.relation_name][0]
        if not self.relation:
            # TODO: defer caller in this case (probably just fired too soon).
            raise LockNoRelationError()

        self.unit = unit or manager.model.unit
        self.app = manager.model.app

    def request(self, callback_id: str, kwargs: dict, max_retry: int | None = -1):
        """Request a lock."""
        q = OperationQueue.from_string(self.relation.data[self.unit].get("operations", ""))
        op_max_retry = -1 if max_retry is None else max_retry
        q.enqueue_request(callback_id, kwargs, max_retry=op_max_retry)
        self.relation.data[self.unit].update({"state": LockIntent.REQUEST.value})
        self.relation.data[self.unit].update({"operations": q.to_string()})

    def retry(self):
        """Grant a lock to a unit."""

        if self.is_max_retry_reached():
            logger.info("Operation max retry reached. Operation complete")
            self.complete()
            return
        self.relation.data[self.unit].update({"last_retry": now_timestamp()})
        self.relation.data[self.unit].update({"state": LockIntent.RETRY.value})
        self.increase_attempt()

    def complete(self):
        """Request to be released"""
        self.relation.data[self.unit].update({"state": LockIntent.IDLE.value})
        q = OperationQueue.from_string(self.relation.data[self.unit].get("operations", ""))
        q.dequeue()
        next_op = q.peek()

        if next_op:
            self.relation.data[self.unit].update({"state": LockIntent.REQUEST.value})
        else:
            self.relation.data[self.unit].update({"state": LockIntent.IDLE.value})

        self.relation.data[self.unit].update({"last_retry": ""})
        self.relation.data[self.unit].update({"attempt": ""})
        self.relation.data[self.unit].update({"operations": q.to_string()})
        self.relation.data[self.unit].update({"completed_at": now_timestamp()})


    def get_operation(self) -> Operation | None:
        q = OperationQueue.from_string(self.relation.data[self.unit].get("operations", ""))
        return q.peek()
    
    def is_max_retry_reached(self) -> bool:
        operation = self.get_operation()
        if not operation:
            return True
        if operation.max_retry == -1:
            return False
        attempt = self.relation.data[self.unit].get("attempt", "")
        attempt_int = int(attempt) if attempt else 0
        return attempt_int >= operation.max_retry
        
    def increase_attempt(self) -> None:
        attempt = self.relation.data[self.unit].get("attempt", "")
        attempt_int = int(attempt) + 1 if attempt else 0
        self.relation.data[self.unit].update({"attempt": str(attempt_int)})

    def attempt(self) -> int:
        attempt = self.relation.data[self.unit].get("attempt", "")
        return attempt if attempt else 0

    def release(self):
        """Unset a lock."""
        self.relation.data[self.app].update({"granted_unit": ""})
        self.relation.data[self.app].update({"granted_at": ""})

    def grant(self):
        """Grant a lock to a unit."""
        self.relation.data[self.app].update({"granted_unit": str(self.unit)})
        self.relation.data[self.app].update({"granted_at": now_timestamp()})

    def is_held(self):
        """This unit holds the lock."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit =  self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.REQUEST.value and granted_unit == str(self.unit)
    
    def is_granted(self):
        granted_unit =  self.relation.data[self.app].get("granted_unit", "")
        return granted_unit == str(self.unit)

    def is_waiting(self):
        """Is this unit waiting for a lock?"""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit =  self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.REQUEST.value and granted_unit != str(self.unit)

    def is_completed(self):
        """A unit has reported that they are finished with the lock."""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit =  self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.IDLE.value and granted_unit == str(self.unit)

    def is_retry(self):
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit =  self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.RETRY.value and granted_unit == str(self.unit)
      
    def is_waiting_retry(self):
        """Is this unit waiting for a lock?"""
        unit_intent = self.relation.data[self.unit].get("state")
        granted_unit =  self.relation.data[self.app].get("granted_unit", "")
        return unit_intent == LockIntent.RETRY.value and granted_unit != str(self.unit)
      


class Locks:
    """Generator that returns a list of locks."""

    def __init__(self, manager):
        self.manager = manager

        # Gather all the units.
        relation = manager.model.relations[manager.relation_name][0]
        units = list(relation.units)

        # Plus our unit ...
        units.append(manager.model.unit)

        self.units = units

    def __iter__(self):
        """Yields a lock for each unit we can find on the relation."""
        for unit in self.units:
            yield Lock(self.manager, unit=unit)


class RunWithLock(EventBase):
    """Event to signal that this unit should run the callback."""

    pass


class AcquireLock(EventBase):
    """Signals that this unit wants to acquire a lock."""

    def __init__(self, handle, callback_id: str, kwargs: dict[str, any] = {}, max_retry: int | None =  None):
        super().__init__(handle)
        self.callback_id = callback_id
        self.kwargs = kwargs
        self.max_retry = max_retry

    def snapshot(self):
        """Snapshot of lock event."""
        return {"callback_id": self.callback_id, "kwargs": self.kwargs, "max_retry": self.max_retry}

    def restore(self, snapshot):
        """Restores lock event."""
        self.callback_id = snapshot["callback_id"]
        self.kwargs = snapshot["kwargs"]
        self.max_retry = snapshot["max_retry"]


class ProcessLocks(EventBase):
    """Used to tell the leader to process all locks."""

    pass

class RollingOpGrantedEvent(EventBase):
    """Custom event emitted when the background worker grants the lock."""


def parse_ts(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, TIMESTAMP_FORMAT)
    except Exception:
        return datetime.now(timezone.utc)

def pick_oldest_retry(locks: list["Lock"]) -> Optional["Lock"]:
    """ Choose the retry lock with the oldest last_retry timestamp."""
    oldest_lock = None
    oldest_ts = None

    for lock in locks:
        ts_str = lock.relation.data[lock.unit].get("last_retry", "")
        ts = parse_ts(ts_str)

        if oldest_ts is None or ts < oldest_ts:
            oldest_ts = ts
            oldest_lock = lock

    return oldest_lock

def pick_oldest_request(locks: list["Lock"]) -> Optional["Lock"]:
    """ Choose the retry lock with the oldest last_retry timestamp."""
    oldest_lock = None
    oldest_timestamp = None

    for lock in locks:
        operation = lock.get_operation()
        if not operation:
            continue
        timestamp = parse_ts(operation.request_time)

        if oldest_timestamp is None or timestamp < oldest_timestamp:
            oldest_timestamp = timestamp
            oldest_lock = lock

    return oldest_lock

def grant_is_after_unit_retry(lock: "Lock") -> bool:
    if not lock.is_retry():
        return True

    grant_ts = parse_ts(lock.relation.data[lock.app].get("granted_at", ""))
    retry_ts = parse_ts(lock.relation.data[lock.unit].get("last_retry", ""))

    # If there was never a retry, treat grant as "after"
    if grant_ts is None:
        return False
    if retry_ts is None:
        return True

    return grant_ts > retry_ts

def grant_is_after_unit_completed(lock: "Lock") -> bool:
    if not lock.is_held():
        return True

    grant_ts = parse_ts(lock.relation.data[lock.app].get("granted_at", ""))
    completed_ts = parse_ts(lock.relation.data[lock.unit].get("completed_at", ""))
    logger.info(f"grant_ts {grant_ts} completed_ts {completed_ts}")

    # If there was never a retry, treat grant as "after"
    if grant_ts is None:
        return False
    if completed_ts is None:
        return True
    
    logger.info(f"grant_ts > completed_ts = {grant_ts > completed_ts}")
    return grant_ts > completed_ts

class RollingOpsManager(Object):
    """Emitters and handlers for rolling ops."""

    #on = RollingOpsCharmEvents()

    def __init__(self, charm: CharmBase, relation: AnyStr, callback_targets = {}):
        """Register our custom events.

        params:
            charm: the charm we are attaching this to.
            relation: an identifier, by convention based on the name of the relation in the
                metadata.yaml, which identifies this instance of RollingOperatorsFactory,
                distinct from other instances that may be handling other events.
            callback: a closure to run when we have a lock. (It must take a CharmBase object and
                EventBase object as args.)
        """
        # "Inherit" from the charm's class. This gives us access to the framework as
        # self.framework, as well as the self.model shortcut.
        super().__init__(charm, None)

        self.relation_name = relation
        self.callback_targets = {"charm":  charm} if not callback_targets else callback_targets

        self.charm = charm  # Maintain a reference to charm, so we can emit events.

        charm.on.define_event("{}_run_with_lock".format(self.relation_name), RunWithLock)
        charm.on.define_event("{}_acquire_lock".format(self.relation_name), AcquireLock)
        #charm.on.define_event("{}_process_locks".format(self.relation_name), ProcessLocks)
        charm.on.define_event("{}_rollingop_granted".format(self.relation_name), RollingOpGrantedEvent)

        # Watch those events (plus the built in relation event).
        self.framework.observe(charm.on[self.relation_name].relation_changed, self._on_relation_changed)
        self.framework.observe(charm.on[self.relation_name].acquire_lock, self._on_acquire_lock)
        #self.framework.observe(charm.on[self.relation_name].run_with_lock, self._on_run_with_lock)
        self.framework.observe(charm.on.leader_elected, self._on_process_locks)
        self.framework.observe(charm.on[self.relation_name].rollingop_granted, self._on_collect_status)
        #self.framework.observe(charm.on.update_status, self._on_collect_status)
        self.framework.observe(charm.on[self.relation_name].relation_departed, self._on_relation_departed)

    def _on_relation_departed(self, event: RelationDepartedEvent):
        if not self.model.unit.is_leader():
            return
        if unit := event.departing_unit:
            lock = Lock(self, unit)
            if lock.is_held():
                lock.release()
        
    def _on_collect_status(self, event):
        if not self.model.unit.is_leader():
            return
        
        relation = self.model.get_relation(self.relation_name)
        if not relation:
            return

        logger.info("THIS IS A DISPATCHED HOOK")
        
        self._on_process_locks()
        
        lock = Lock(self)
        if lock.is_held():
            self._on_run_with_lock()

    def _on_relation_changed(self: CharmBase, event: RelationChangedEvent):
        """Process relation changed.

        First, determine whether this unit has been granted a lock. If so, emit a RunWithLock
        event.

        Then, if we are the leader, fire off a process locks event.

        """
        lock = Lock(self)

        if lock.is_waiting():  # REQUEST - none
            self.model.unit.status = WaitingStatus("Awaiting {} operation".format(self.relation_name))

        if lock.is_held(): # REQUEST - GRANTED
            logger.info(f"it's REQUEST - GRANTED")
            self._on_run_with_lock()

        if lock.is_retry() and grant_is_after_unit_retry(lock): # RETRY - GRANTED
            logger.info(f"it's RETRY - GRANTED")
            self._on_run_with_lock()

        if self.model.unit.is_leader():
            self._on_process_locks()


    def _on_process_locks(self, event: ProcessLocks = None):
        """Process locks.

        Runs only on the leader. Updates the status of all locks.

        """
        if not self.model.unit.is_leader():
            return

        for lock in Locks(self):
            if lock.is_completed():  # IDLE - granted 
                logger.info(f"{ lock.unit} released after release")
                lock.release() # IDLE - none

            elif lock.is_retry() and not grant_is_after_unit_retry(lock): # RETRY - granted
                logger.info(f"{ lock.unit} released after retry")
                lock.release()

            elif lock.is_held() and not grant_is_after_unit_completed(lock): # REQUEST - granted
                logger.info(f"{ lock.unit} released after completed (other operation waiting)")
                lock.release()

        relation = self.model.get_relation(self.relation_name)
        granted_unit = relation.data[self.model.app].get("granted_unit", "")
        logger.info(f"{ granted_unit} granted")
        if granted_unit:
            logger.info(f"{ granted_unit} already scheduled")
            return # REQUEST - GRANTED -> running get out
        
        self.schedule()

    def schedule(self) -> None:
        logger.info("starting scheduling")
        
        g1_request = []
        g2_retry = []

        for lock in Locks(self):
            unit_state = lock.relation.data[lock.unit].get("state", "")
            granted_unit = lock.relation.data[lock.app].get("granted_unit", "")
            
            logger.info(f"PROCESSING {lock.unit} unit={unit_state} app={granted_unit}")

            if lock.is_waiting(): # REQUEST - none
                g1_request.append(lock)

            elif lock.is_waiting_retry(): # RETRY - none 
                g2_retry.append(lock)

        logger.info(f"g1_request {g1_request}")
        logger.info(f"g2_retry {g2_retry}")

        selected = None
        if g1_request:
            selected = pick_oldest_request(g1_request)
        elif g2_retry:
            selected = pick_oldest_retry(g2_retry)
        
        if not selected:
            self.model.app.status = ActiveStatus()
            return
        
        selected.grant()
        if selected.unit == self.model.unit:
            self._on_run_with_lock() # REQUEST - granted 
            return

    def _on_acquire_lock(self: CharmBase, event: AcquireLock):
        """Request a lock."""
        try:
            lock = Lock(self)
            lock.request(event.callback_id, event.kwargs, event.max_retry)

            if self.model.unit.is_leader():
                relation = self.model.get_relation(self.relation_name)
                self.charm.on[self.relation_name].relation_changed.emit(relation, app=self.charm.app)

        except LockNoRelationError:
            logger.debug("No {} peer relation yet. Delaying rolling op.".format(self.relation_name))
            event.defer()


    def _on_run_with_lock(self: CharmBase):
        lock = Lock(self)
        self.model.unit.status = MaintenanceStatus("Executing {} operation".format(self.relation_name))
        operation = lock.get_operation()
        if not operation:
            lock.complete()
            if self.model.unit.is_leader():
                self.model.unit.status = ActiveStatus()
                self._on_process_locks()
                return
        
        callback = self.callback_targets.get(operation.callback_id)

        logger.info(f"executing {lock.attempt()}, callback_id {operation.callback_id}, {callback}")
        kwargs = json.loads(operation.kwargs) if operation.kwargs else {}

        try:
            result = callback(**kwargs)
        except Exception as e:
            logger.error(f"{e}")
            result = OperationResult.RETRY   
        
        if result == OperationResult.RETRY:
            lock.retry()
            self.model.unit.status = MaintenanceStatus("Rolling will be retried {}".format(self.relation_name))
            if self.model.unit.is_leader():
                self._on_process_locks()
            return

        lock.complete()
        if self.model.unit.is_leader():
            self.model.unit.status = ActiveStatus()
            self._on_process_locks()
            return

        if self.model.unit.status.message == f"Executing {self.relation_name} operation":
            self.model.unit.status = ActiveStatus()
