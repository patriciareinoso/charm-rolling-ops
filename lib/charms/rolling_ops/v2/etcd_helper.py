import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import argparse
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from sys import version_info
from typing import Any, Optional

@dataclass
class Operation:
    """A single queued operation."""

    callback_id: str
    kwargs: dict[str, Any]
    requested_at: Optional[datetime]
    max_retry: Optional[int]
    attempt: int
    result: str

    def __post_init__(self) -> None:
        """Vallidate the class attributes."""
        if not isinstance(self.callback_id, str) or not self.callback_id.strip():
            raise ValueError("callback_id must be a non-empty string")

        if not isinstance(self.kwargs, dict):
            raise ValueError("kwargs must be a dict")
        try:
            json.dumps(self.kwargs)
        except TypeError as e:
            raise ValueError(f"kwargs must be JSON-serializable: {e}") from e

        if self.requested_at is not None and not isinstance(self.requested_at, datetime):
            raise ValueError("requested_at must be a datetime or None")

        if self.max_retry:
            if not isinstance(self.max_retry, int):
                raise ValueError("max_retry must be an int")
            if self.max_retry < 0:
                raise ValueError("max_retry must be >= 0")

        if not isinstance(self.attempt, int):
            raise ValueError("attempt must be an int")
        if self.attempt < 0:
            raise ValueError("attempt must be >= 0")

    @classmethod
    def create(
        cls,
        callback_id: str,
        kwargs: dict[str, Any],
        max_retry: int | None = None,
    ) -> "Operation":
        """Create a new operation from a callback id and kwargs."""
        return cls(
            callback_id=callback_id,
            kwargs=kwargs,
            requested_at=_now_timestamp(),
            max_retry=max_retry,
            attempt=0,
            result=""
        )

    def _to_dict(self) -> dict[str, str]:
        """Dict form (string-only values)."""
        return {
            "callback_id": self.callback_id,
            "kwargs": _args_to_json(self.kwargs),
            "requested_at": self.requested_at.strftime(TIMESTAMP_FORMAT)
            if self.requested_at
            else "",
            "max_retry": str(self.max_retry) if self.max_retry else "",
            "attempt": str(self.attempt),
            "result" : self.result,
        }

    def to_string(self) -> str:
        """Serialize to a string suitable for a Juju databag."""
        return json.dumps(self._to_dict(), separators=(",", ":"))
    
    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Operation":
        """Create an Operation from its dict (etcd) representation."""
        try:
            requested_at = (
                datetime.strptime(data["requested_at"], TIMESTAMP_FORMAT)
                if data.get("requested_at")
                else None
            )

            max_retry = (
                int(data["max_retry"])
                if data.get("max_retry")
                else None
            )

            return cls(
                callback_id=data["callback_id"],
                kwargs=json.loads(data["kwargs"]) if data.get("kwargs") else {},
                requested_at=requested_at,
                max_retry=max_retry,
                attempt=int(data["attempt"]),
                result=data.get("result", ""),
            )

        except KeyError as e:
            raise ValueError(f"Missing required field: {e}") from e
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid Operation dict: {e}") from e

    def increase_attempt(self) -> None:
        """Increment the attempt counter."""
        self.attempt += 1

    def is_max_retry_reached(self) -> bool:
        """Return True if attempt exceeds max_retry (unless max_retry is None)."""
        if not self.max_retry:
            return False
        return self.attempt > self.max_retry
    
    @property
    def op_id(self):
        return f"{self.requested_at}-{self.callback_id}"

@dataclass(frozen=True)
class Keys:
    base: str         # /rollingops/<owner>
    lock_key: str     # /rollingops/granted-unit
    pending: str      # <base>/pending/
    inprogress: str   # <base>/inprogress/
    completed: str    # <base>/comp
    
def make_keys(owner: str) -> Keys:
    base = f"/rollingops/{owner}"
    return Keys(
        base=base,
        lock_key=f"/rollingops/granted-unit",
        pending=f"{base}/pending/",
        inprogress=f"{base}/inprogress/",
        completed=f"{base}/completed/",
    )

# ---------- helpers ----------

def sh(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)

def etcdctl(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return sh(["etcdctl", *args], check=check, capture=capture)

def etcd_get_json(key: str) -> Optional[dict[str, Any]]:
    res = etcdctl([
        "get",
        key,
        "--print-value-only",
    ], check=True)

    if not res.stdout.strip():
        return {}

    out = res.stdout.splitlines()
    return json.loads(out[0]) if out else None

def etcd_get_first_key(key_prefix: str) -> Optional[str]:
    res = etcdctl(["get", key_prefix, "--prefix", "--keys-only", "--limit=1"], check=False)
    if res.returncode != 0:
        return None
    out = res.stdout.strip().splitlines()
    return out[0] if out else None

def etcd_get_last_key(key_prefix: str) -> Optional[str]:
    res = etcdctl(["get", key_prefix, "--prefix", "--keys-only", "--sort-by=KEY", "--order=DESCEND", "--limit=1"], check=False)
    if res.returncode != 0:
        return None
    out = res.stdout.strip().splitlines()
    return out[0] if out else None


# ---------- key helpers ----------
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_timestamp_str() -> str:
    """UTC timestamp string with microseconds."""
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def _now_timestamp() -> datetime:
    """UTC timestamp string with microseconds."""
    return datetime.now(timezone.utc)


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse timestamp string. Return 'now' on errors to avoid selecting invalid timestamps."""
    try:
        dt = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _args_to_json(data: dict[str, Any]) -> str:
    """Deterministic JSON serialization for kwargs."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def cleanup_completed(keys: Keys, owner: str) -> None:
    completed_key = etcd_get_first_key(keys.completed)
    if not completed_key:
        return False

    txn = f"""\
    value("{keys.lock_key}") = "{owner}"
    version("{completed_key}") != "0"

    del f"{completed_key}"


    """
    res = sh(["bash", "-lc", f"printf %s '{txn}' | etcdctl txn"], check=False)
    return "SUCCESS" in res.stdout

def move_operation(from_queue: str, to_queue: str, lock_key: str, owner: str) -> bool:
    head = etcd_get_first_key(from_queue)
    if not head:
        return False

    opid = head.split("/")[-1]
    new_key = f"{to_queue}{opid}"

    value = etcd_get_json(head)
    data = json.dumps(value)
    value_escaped = data.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    txn = f"""\
    version("{head}") != "0"

    put "{new_key}" "{value_escaped}"
    del "{head}"


    """
    res = sh(["bash", "-lc", f"printf %s '{txn}' | etcdctl txn"], check=False)
    return "SUCCESS" in res.stdout


def get_lease(ttl: int) -> str:
    """ Create a lease and return its ID."""
    res = etcdctl(["lease", "grant", str(ttl)])
    # parse: "lease 694d9c9aeca3422a granted with TTL(1800s)"
    parts = res.stdout.strip().split()
    return parts[1]

def lease_keepalive_once(lease_id: str) -> None:
    print(etcdctl(["lease", "keep-alive", lease_id, "--once"], check=False))

def release_lock(keys: Keys, owner: str) -> None:
    etcdctl(["del", keys.lock_key], check=False)

    txn = f"""\
    value("{keys.lock_key}") = "{owner}"

    del "{keys.lock_key}"


    """
    res = sh(["bash", "-lc", f"printf %s '{txn}' | etcdctl txn"], check=False)
    return "SUCCESS" in res.stdout

def etcd_get_operation(key: str) -> Optional[Operation]:
    res = etcd_get_json(key)
    if not res:
        return None
    return Operation.from_dict(res)

def try_acquire_lock(keys: Keys, owner: str, lease_id: str) -> bool:
    txn = f"""\
    version("{keys.lock_key}") = "0"

    put "{keys.lock_key}" "{owner}" --lease={lease_id}


    """
    res = sh(["bash", "-lc", f"printf %s '{txn}' | etcdctl txn"], check=False)
    return "SUCCESS" in res.stdout


def watch_queue(key_prefix: str):
    proc = subprocess.Popen(
        ["etcdctl", "watch", key_prefix, "--prefix", "--write-out=json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        event = json.loads(proc.stdout.readline())
        return event
    finally:
        proc.terminate()
        proc.wait(timeout=1)

def start_lease_keepalive(lease_id: str) -> str:
    return subprocess.Popen(
        ["etcdctl", "lease", "keep-alive", lease_id],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    ).pid

def stop_keepalive(pid: str) -> None:
    try:
        os.kill(pid, signal.SIGINT)
    except OSError:
        pass

def revoke_lease(lease_id: str):
    etcdctl(["etcdctl", "lease", "revoke", lease_id])


def finish_execution(keys: Keys, owner: str, lease_id: str, pid: str):
    stop_keepalive(pid)
    release_lock(keys, owner)
    revoke_lease(lease_id)

def put_operation(key_prefix: str, operation: Operation):

    op_str = operation.to_string()
    op_id = f"{operation.requested_at}-{operation.callback_id}"
    key = f"{key_prefix}{op_id}"
    etcdctl(["put", key, op_str])


def main():
    owner = "model.unit1"
    keys = make_keys( owner)
    operation = Operation.create("restart", {}, 3)
    put_operation(keys.pending, operation)
    move_operation(keys.pending, keys.inprogress, keys.lock_key, owner)
    #op_id = f"{operation.requested_at}-{operation.callback_id}"
    #key = f"{keys.pending}{op_id}"
    #print(etcd_get_operation(key))



def function2():
#def main():
    res = etcdctl(["version"])
    print(res.stdout)

    # etcdctl get /rollingops/cluster1/granted-unit
    # etcdctl get /rollingops/cluster1/model.unit1/pending/1-operation
    # etcdctl put /rollingops/cluster1/model.unit1/pending/1-operation "1234"
    # etcdctl get /rollingops/cluster1/model.unit1/inprogress/1-operation
    # etcdctl del /rollingops/cluster1/model.unit1/inprogress/1-operation
    # etcdctl put /rollingops/cluster1/model.unit1/completed/1-operation "1234"
    owner = "model.unit1"
    cluster_id = "cluster1" 
    keys = make_keys(cluster_id, owner)

    lease_id = None
    holding_lock = False
    lock_lease_ttl = 60
    acquire_retry_sleep = 10
    attempt = 0

    lease_id = get_lease(lock_lease_ttl)
    success = try_acquire_lock(keys, owner, lease_id)
    print(success)
    pid=0

    while True:
        print(f"attempt {attempt}")
        if pending_key := etcd_get_first_key(keys.pending) and not pending_key:
            time.sleep(acquire_retry_sleep)
            continue
        if not holding_lock:

            if lease_id is None:
                lease_id = get_lease(lock_lease_ttl)
                pid = start_lease_keepalive(lease_id)

            if try_acquire_lock(keys, owner, lease_id):
                holding_lock = True
            else:
                attempt += 1
                time.sleep(acquire_retry_sleep)
                continue
        
        moved = move_operation(keys.pending, keys.inprogress, keys.lock_key, owner)
        if moved:
            # dispatch hook
            print("dispatch hook")
        else:
            time.sleep(acquire_retry_sleep)

        print(watch_queue(keys.completed))
        completed_key = etcd_get_first_key(keys.completed)
        operation = etcd_get_operation(completed_key)

        if operation.result == "retry-hold" and not operation.is_max_retry_reached():
            move_operation(keys.completed, keys.inprogress, keys.lock_key, owner)
            # dispatch hook
            print("dispatch hook")
            continue

        elif operation.result == "retry-release" and not operation.is_max_retry_reached():
            move_operation(keys.completed, keys.pending, keys.lock_key, owner)

        else:
            cleanup_completed(keys)

        finish_execution(keys,owner,lease_id, pid)
        holding_lock = False



if __name__ == "__main__":
    main()