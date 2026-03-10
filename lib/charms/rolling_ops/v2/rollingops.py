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

"""Rolling Ops v1 — coordinated rolling operations for Juju charms.

This library provides a reusable mechanism for coordinating rolling operations
across units of a Juju application using a peer-relation distributed lock.

The library guarantees that at most one unit executes a rolling operation at any
time, while allowing multiple units to enqueue operations and participate
in a coordinated rollout.

## Data model (peer relation)

### Unit databag

Each unit maintains a FIFO queue of operations it wishes to execute.

Keys:
- `operations`: JSON-encoded list of queued `Operation` objects
- `state`: `"idle"` | `"request"` | `"retry"`
- `executed_at`: UTC timestamp string indicating when the current operation last ran

Each `Operation` contains:
- `callback_id`: identifier of the callback to execute
- `kwargs`: JSON-serializable arguments for the callback
- `requested_at`: UTC timestamp when the operation was enqueued
- `max_retry`: maximum retry count (`< 0` means unlimited)
- `attempt`: current attempt number

### Application databag

The application databag represents the global lock state.

Keys:
- `granted_unit`: unit identifier (unit name), or empty
- `granted_at`: UTC timestamp indicating when the lock was granted

## Operation semantics

- Units enqueue operations instead of overwriting a single pending request.
- Duplicate operations (same `callback_id` and `kwargs`) are ignored if they are
  already the last queued operation.
- When granted the lock, a unit executes exactly one operation (the queue head).
- After execution, the lock is released so that other units may proceed.

## Retry semantics

- If a callback returns `OperationResult.RETRY_RELEASE` the unit will release the
lock and retry the operation later.
- If a callback return `OperationResult.RETRY_HOLD` the unit will keep the
lock and retry immediately.
- Retry state (`attempt`) is tracked per operation.
- When `max_retry` is exceeded, the failing operation is dropped and the unit
  proceeds to the next queued operation, if any.

## Scheduling semantics

- Only the leader schedules lock grants.
- If a valid lock grant exists, no new unit is scheduled.
- Requests are preferred over retries.
- Among requests, the operation with the oldest `requested_at` timestamp is selected.
- Among retries, the operation with the oldest `executed_at` timestamp is selected.
- Stale grants (e.g., pointing to departed units) are automatically released.

All timestamps are stored in UTC using `TIMESTAMP_FORMAT`.

## Using the library in a charm

### 1. Declare a peer relation

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
from charms.rolling_ops.v1.rollingops import RollingOpsManagerv1, OperationResult

class SomeCharm(CharmBase):
    def __init__(self, *args):
        super().__init__(*args)

        self.rolling_ops = RollingOpsManagerv1(
            charm=self,
            relation="restart",
            callback_targets={
                "restart": self._restart,
                "failed_restart": self._failed_restart,
                "defer_restart": self._defer_restart,
            },
        )

    def _restart(self, force: bool) -> OperationResult:
        # perform restart logic
        return OperationResult.RELEASE

    def _failed_restart(self) -> OperationResult:
        # perform restart logic
        return OperationResult.RETRY_RELEASE

    def _defer_restart(self) -> OperationResult:
        if not self.ready():
            event.defer()
            return OperationResult.RETRY_HOLD
        # do restart logic
        return OperationResult.RELEASE
```

Request a rolling operation

```python

    def _on_restart_action(self, event):
        self.rolling_ops.request_async_lock(
            callback_id="restart",
            kwargs={"force": True},
            max_retry=3,
    )
```

All participating units must enqueue the operation in order to be included
in the rolling execution.

Units that do not enqueue the operation will be skipped, allowing operators
to recover from partial failures by reissuing requests selectively.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from sys import version_info
from typing import Any, Optional

from charmlibs.interfaces.tls_certificates import (
    generate_ca,
    generate_certificate,
    generate_csr,
    generate_private_key,
)
from charms.data_platform_libs.v0.data_interfaces import EtcdRequires
from ops import Relation
from ops.charm import (
    CharmBase,
    RelationDepartedEvent,
)
from ops.framework import EventBase, Object
from tenacity import retry, stop_after_delay, wait_fixed

logger = logging.getLogger(__name__)

# The unique Charmhub library identifier, never change it
LIBID = "20b7777f58fe421e9a223aefc2b4d3a4"

# Increment this major API version when introducing breaking changes
LIBAPI = 1

# Increment this PATCH version before using `charmcraft publish-lib` or reset
# to 0 if you are raising the major API version
LIBPATCH = 0

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
SECRET_FIELD = "rollingops-client-secret-id"


def _now_timestamp() -> datetime:
    """UTC timestamp string with microseconds."""
    return datetime.now(timezone.utc)


def _args_to_json(data: dict[str, Any]) -> str:
    """Deterministic JSON serialization for kwargs."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


class LockNoRelationError(Exception):
    """Raised if we are trying to process a lock, but do not appear to have a relation yet."""


class EtcdUnreachableError(Exception):
    """Raised if etcd server is unreachable."""


class EtcdNotConfiguredError(Exception):
    """Raised if etcd client has not been configured yet (env file does not exist)."""


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
            result="",
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
            "result": self.result,
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

            max_retry = int(data["max_retry"]) if data.get("max_retry") else None

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

    def complete(self) -> None:
        """Mark the operation as completed to indicate the lock should be released."""
        self.increase_attempt()
        self.result = OperationResult.RELEASE.value

    def retry_release(self) -> None:
        """Mark the operation for retry if it has not reached the max retry."""
        self.increase_attempt()
        if self.is_max_retry_reached():
            self.result = OperationResult.RELEASE.value
        else:
            self.result = OperationResult.RETRY_RELEASE.value

    def retry_hold(self) -> None:
        """Mark the operation for retry if it has not reached the max retry."""
        self.increase_attempt()
        if self.is_max_retry_reached():
            self.result = OperationResult.RELEASE.value
        else:
            self.result = OperationResult.RETRY_HOLD.value

    @property
    def op_id(self):
        """Return the unique identifier for this operation."""
        return f"{self.requested_at.strftime(TIMESTAMP_FORMAT)}-{self.callback_id}"

    def __eq__(self, other: object) -> bool:
        """Equal for the operation."""
        if not isinstance(other, Operation):
            return NotImplemented
        return self.callback_id == other.callback_id and self.kwargs == other.kwargs

    def __hash__(self) -> int:
        """Hash for the operation."""
        return hash((self.callback_id, _args_to_json(self.kwargs)))


@dataclass(frozen=True)
class Keys:
    """Collection of etcd key prefixes used for rolling operations.

    This dataclass defines the etcd key structure used by the rolling
    operations system. Each unit (owner) has its own set of queues
    under a dedicated base prefix, while the lock key is shared
    across all units.

    Attributes:
        base: Root prefix for all queues owned by a specific unit.
        lock_key: Global key used to represent the distributed lock owner.
    """

    base: str
    lock_key: str = "/rollingops/granted-unit"

    @property
    def pending(self) -> str:
        """Prefix for operations waiting to be executed."""
        return f"{self.base}/pending/"

    @property
    def inprogress(self) -> str:
        """Prefix for operations currently being executed."""
        return f"{self.base}/inprogress/"

    @property
    def completed(self) -> str:
        """Prefix for operations that have finished execution."""
        return f"{self.base}/completed/"

    @classmethod
    def for_owner(cls, owner: str) -> "Keys":
        """Create a set of keys for a given owner."""
        return cls(base=f"/rollingops/{owner}")


class OperationResult(Enum):
    """Callback return values."""

    RELEASE = "release"
    RETRY_RELEASE = "retry-release"
    RETRY_HOLD = "retry-hold"


class RollingOpsLockGrantedEvent(EventBase):
    """Custom event emitted when the background worker grants the lock."""


class CertificatesManager:
    """Manage generation and persistence of TLS certificates for etcd client access.

    This class is responsible for creating and storing a client Certificate
    Authority (CA) and a client certificate/key pair used to authenticate
    with etcd via TLS. Certificates are generated only once and persisted
    under a local directory so they can be reused across charm executions.

    Certificates are not renewed or rotated.

    Args:
        base_dir: Directory where certificates and keys will be stored.
        validity_days: Number of days the generated certificates remain valid.
    """

    BASE_DIR = Path("/var/lib/rollingops/tls")

    CA_KEY = BASE_DIR / "client-ca.key"
    CA_CERT = BASE_DIR / "client-ca.pem"
    CLIENT_KEY = BASE_DIR / "client.key"
    CLIENT_CERT = BASE_DIR / "client.pem"

    VALIDITY_DAYS = 365 * 10

    @classmethod
    def exists(cls) -> bool:
        """Check whether the required client certificates already exist.

        Returns:
            True if the client certificate and key and the CA certificate
            are present on disk, otherwise False.
        """
        return (
            cls.CA_KEY.exists()
            and cls.CA_CERT.exists()
            and cls.CLIENT_KEY.exists()
            and cls.CLIENT_CERT.exists()
        )

    @classmethod
    def load_client_cert_and_key(cls) -> tuple[str, str]:
        """Load the client certificate and private key from disk.

        Returns:
            A tuple containing:
            - The client certificate PEM string
            - The client private key PEM string
        """
        return cls.CLIENT_CERT.read_text(), cls.CLIENT_KEY.read_text()

    @classmethod
    def client_paths(cls) -> tuple[Path, Path]:
        """Return filesystem paths for the client certificate and key.

        Returns:
            A tuple containing:
            - Path to the client certificate
            - Path to the client private key
        """
        return cls.CLIENT_CERT, cls.CLIENT_KEY

    @classmethod
    def persist_client_cert_and_key(cls, cert_pem: str, key_pem: str) -> None:
        """Persist the provided client certificate and key to disk.

        Args:
            cert_pem: PEM-encoded client certificate.
            key_pem: PEM-encoded client private key.
        """
        cls.CLIENT_CERT.write_text(cert_pem)
        cls.CLIENT_KEY.write_text(key_pem)

        os.chmod(cls.CLIENT_CERT, 0o644)
        os.chmod(cls.CLIENT_KEY, 0o600)

    @classmethod
    def has_client_cert_and_key(cls, cert_pem: str, key_pem: str) -> bool:
        """Return whether the provided certificate material matches local files."""
        if not cls.CLIENT_CERT.exists() or not cls.CLIENT_KEY.exists():
            return False

        return cls.CLIENT_CERT.read_text() == cert_pem and cls.CLIENT_KEY.read_text() == key_pem

    @classmethod
    def generate(cls, common_name: str) -> None:
        """Generate a client CA and client certificate if they do not exist.

        This method creates:
        1. A CA private key and self-signed CA certificate.
        2. A client private key.
        3. A certificate signing request (CSR) using the provided common name.
        4. A client certificate signed by the generated CA.

        The generated files are written to disk and reused in future runs.
        If the certificates already exist, this method does nothing.

        Args:
            common_name: Common Name (CN) used in the client certificate
                subject. This value should not contain slashes.
        """
        if cls.exists():
            return

        ca_key = generate_private_key(key_size=4096)
        ca_crt = generate_ca(
            private_key=ca_key,
            common_name="rollingops-client-ca",
            validity=timedelta(days=cls.VALIDITY_DAYS),
        )

        client_key = generate_private_key(key_size=4096)

        csr = generate_csr(
            private_key=client_key,
            common_name=common_name,
            add_unique_id_to_subject_name=False,
        )

        client_crt = generate_certificate(
            csr=csr,
            ca=ca_crt,
            ca_private_key=ca_key,
            validity=timedelta(days=cls.VALIDITY_DAYS),
            is_ca=False,
        )

        cls.CA_KEY.write_text(ca_key.raw)
        cls.CA_CERT.write_text(ca_crt.raw)
        cls.CLIENT_KEY.write_text(client_key.raw)
        cls.CLIENT_CERT.write_text(client_crt.raw)

        os.chmod(cls.CA_KEY, 0o600)
        os.chmod(cls.CLIENT_KEY, 0o600)
        os.chmod(cls.CA_CERT, 0o644)
        os.chmod(cls.CLIENT_CERT, 0o644)


class EtcdCtl:
    """Class for interacting with etcd through the etcdctl CLI.

    This class encapsulates configuration and execution of the tool. It manages
    the environment variables required for connecting to an etcd cluster,
    including TLS configuration, and provides convenience methods for
    executing commands and retrieving structured results.
    """

    BASE_DIR = Path("/var/lib/rollingops/etcd")
    SERVER_CA = BASE_DIR / "server-ca.pem"
    ENV_FILE = BASE_DIR / "etcdctl.env"

    @classmethod
    def write_env_file(
        cls,
        endpoints: str,
        tls_ca_pem: str,
        client_cert_path: Path,
        client_key_path: Path,
    ) -> None:
        """Create or update the etcdctl environment configuration file.

        This method writes an environment file containing the required
        ETCDCTL_* variables used by etcdctl to connect to the etcd cluster.

        Args:
            endpoints: Comma-separated list of etcd endpoints.
            tls_ca_pem: PEM-encoded CA certificate used to verify the etcd server.
            client_cert_path: Path to the client TLS certificate.
            client_key_path: Path to the client TLS private key.
        """
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.SERVER_CA.write_text(tls_ca_pem or "")
        os.chmod(cls.SERVER_CA, 0o644)

        lines = [
            'export ETCDCTL_API="3"',
            f'export ETCDCTL_ENDPOINTS="{endpoints}"',
            f'export ETCDCTL_CACERT="{cls.SERVER_CA}"',
            f'export ETCDCTL_CERT="{client_cert_path}"',
            f'export ETCDCTL_KEY="{client_key_path}"',
            "",
        ]

        cls.ENV_FILE.write_text("\n".join(lines))
        os.chmod(cls.ENV_FILE, 0o600)

    @classmethod
    def load_env(cls) -> dict[str, str]:
        """Load etcdctl environment variables from the env file.

        Parses the generated environment file and extracts ETCDCTL_*
        variables so they can be injected into subprocess environments.

        Returns:
            A dictionary containing environment variables to pass to
            subprocess calls.

        Raises:
            EtcdNotConfiguredError: If the environment file does not exist.
        """
        cls.ensure_initialized()

        env = os.environ.copy()

        for line in cls.ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if not line.startswith("ETCDCTL_"):
                continue

            key, value = line.split("=", 1)
            env[key] = value.strip().strip('"').strip("'")

        env.setdefault("ETCDCTL_API", "3")
        return env

    @classmethod
    def ensure_initialized(cls):
        """Checks whether the environment file for etcdctl is setup."""
        if not cls.ENV_FILE.exists():
            raise EtcdNotConfiguredError(f"etcdctl env file does not exist: {cls.ENV_FILE}")

    @classmethod
    def run(
        cls, args: list[str], check: bool = True, capture: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute an etcdctl command.

        Args:
            args: List of arguments to pass to etcdctl.
            check: If True, raise an exception on non-zero exit status.
            capture: Whether to capture stdout and stderr.

        Returns:
            A CompletedProcess object containing the result.
        """
        cls.ensure_initialized()
        cmd = ["etcdctl", *args]
        return subprocess.run(
            cmd, env=cls.load_env(), check=check, text=True, capture_output=capture
        )

    @classmethod
    def get_first_key_value(cls, key_prefix: str) -> Optional[tuple[str, dict]]:
        """Retrieve the first key and value under a given prefix.

        Args:
            key_prefix: Key prefix to search for.

        Returns:
            A tuple containing:
            - The key string
            - The parsed JSON value as a dictionary

            Returns None if no key exists or the command fails.
        """
        res = cls.run(
            ["get", key_prefix, "--prefix", "--limit=1"],
            check=False,
        )

        if res.returncode != 0:
            return None

        out = res.stdout.strip().splitlines()
        if len(out) < 2:
            return None

        return out[0], json.loads(out[1])

    @classmethod
    def get_last_key_value(cls, key_prefix: str) -> Optional[tuple[str, dict]]:
        """Retrieve the last key and value under a given prefix.

        Args:
            key_prefix: Key prefix to search for.

        Returns:
            A tuple containing:
            - The key string
            - The parsed JSON value as a dictionary

            Returns None if no key exists or the command fails.
        """
        res = cls.run(
            ["get", key_prefix, "--prefix", "--sort-by=KEY", "--order=DESCEND", "--limit=1"],
            check=False,
        )
        if res.returncode != 0:
            return None
        out = res.stdout.strip().splitlines()
        if len(out) < 2:
            return None

        return out[0], json.loads(out[1])

    @classmethod
    def txn(cls, txn: str) -> bool:
        """Execute an etcd transaction.

        The transaction string should follow the etcdctl transaction format
        where comparison statements are followed by operations.

        Args:
            txn: The transaction specification passed to `etcdctl txn`.

        Returns:
            True if the transaction succeeded, otherwise False.
        """
        cls.ensure_initialized()
        res = subprocess.run(
            ["bash", "-lc", f"printf %s '{txn}' | etcdctl txn"],
            text=True,
            env=cls.load_env(),
            capture_output=True,
            check=False,
        )

        logger.debug("etcd txn result: %s", res.stdout)
        return "SUCCESS" in res.stdout


class EtcdLease:
    """Manage the lifecycle of an etcd lease and its keep-alive process."""

    def __init__(self):
        self.id: str | None = None
        self.keepalive_proc: subprocess.Popen | None = None

    def grant(self, ttl: int) -> None:
        """Create a new lease and start the keep-alive process.

        Args:
            ttl: Time-to-live of the lease in seconds.
        """
        res = EtcdCtl.run(["lease", "grant", str(ttl)])
        # parse: "lease 694d9c9aeca3422a granted with TTL(1800s)"
        parts = res.stdout.strip().split()
        self.id = parts[1]
        self._start_lease_keepalive()

    def revoke(self) -> None:
        """Revoke the current lease and stop the keep-alive process."""
        if self.id is not None:
            EtcdCtl.run(["lease", "revoke", self.id])
            self.id = None
        self._stop_keepalive()

    def _start_lease_keepalive(self) -> None:
        """Start the background process that keeps the lease alive."""
        EtcdCtl.ensure_initialized()
        self.keepalive_proc = subprocess.Popen(
            ["etcdctl", "lease", "keep-alive", self.id],
            env=EtcdCtl.load_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _stop_keepalive(self) -> None:
        """Terminate the keep-alive subprocess if it is running."""
        if self.keepalive_proc is None:
            return
        self.keepalive_proc.terminate()
        try:
            self.keepalive_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.keepalive_proc.kill()
            self.keepalive_proc.wait(timeout=2)
        self.keepalive_proc = None


class EtcdLock:
    """Distributed lock implementation backed by etcd.

    The lock is represented by a key whose value identifies the current owner.

    Lock acquisition and release are performed using transactions to
    ensure atomicity.

    The lock is attached to an etcd lease so that it is
    automatically released if the owner stops refreshing the lease.
    """

    def __init__(self, lock_key: str, owner: str):
        self.lock_key = lock_key
        self.owner = owner

    def try_acquire(self, lease_id: str) -> bool:
        """Attempt to acquire the lock.

        This method uses an etcd transaction that succeeds only if the
        lock key does not yet exist. If successful, the lock key is created with the current
        owner as its value and is attached to the provided lease.

        Args:
            lease_id: ID of the etcd lease to associate with the lock.

        Returns:
            True if the lock was successfully acquired, otherwise False.
        """
        txn = f"""\
        version("{self.lock_key}") = "0"

        put "{self.lock_key}" "{self.owner}" --lease={lease_id}


        """
        return EtcdCtl.txn(txn)

    def release(self) -> None:
        """Release the lock if it is currently held by this owner.

        The lock is removed only if the value of the lock key matches
        the current owner. This prevents one process from accidentally
        releasing a lock held by another owner.
        """
        txn = f"""\
        value("{self.lock_key}") = "{self.owner}"

        del "{self.lock_key}"


        """
        EtcdCtl.txn(txn)

    def is_held(self) -> bool:
        """Check whether the lock is currently held by this owner."""
        proc = EtcdCtl.run(["get", self.lock_key, "--print-value-only"], check=False)

        if proc.returncode != 0:
            return False

        value = proc.stdout.strip()
        return value == self.owner


class EtcdOperationQueue:
    """Queue abstraction for operations stored in etcd.

    This class represents a queue of operations stored under a common
    key prefix in etcd. Each operation is stored as a key-value pair
    where the key encodes the operation identifier and ordering, and
    the value contains the serialized operation data.
    """

    def __init__(self, prefix: str, lock: EtcdLock):
        self.prefix = prefix
        self.lock = lock

    def peek(self) -> Optional[Operation]:
        """Return the first operation in the queue without removing it."""
        kv = EtcdCtl.get_first_key_value(self.prefix)
        if not kv:
            return None
        _, value = kv
        return Operation.from_dict(value)

    def _peek_last(self) -> Optional[Operation]:
        """Return the last operation in the queue without removing it."""
        kv = EtcdCtl.get_last_key_value(self.prefix)
        if not kv:
            return None
        _, value = kv
        return Operation.from_dict(value)

    def move_head(self, to_queue_prefix: str) -> bool:
        """Move the first operation in the queue to another queue.

        This operation is performed atomically using an etcd transaction.
        The transaction succeeds only if:
        - The lock is currently held by the configured owner.
        - The head operation still exists.

        Args:
            to_queue_prefix: Destination queue prefix.

        Returns:
            True if the operation was moved successfully, otherwise False.
        """
        kv = EtcdCtl.get_first_key_value(self.prefix)
        if not kv:
            return None
        key, value = kv

        op_id = key.split("/")[-1]
        new_key = f"{to_queue_prefix}{op_id}"
        data = json.dumps(value)
        value_escaped = data.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        txn = f"""\
        value("{self.lock.lock_key}") = "{self.lock.owner}"
        version("{key}") != "0"

        put "{new_key}" "{value_escaped}"
        del "{key}"


        """
        return EtcdCtl.txn(txn)

    def move_operation(self, to_queue_prefix: str, operation: Operation) -> bool:
        """Move a specific operation from this queue to another queue.

        The operation is identified using its operation ID and moved
        atomically via an etcd transaction.

        Args:
            to_queue_prefix: Destination queue prefix.
            operation: Operation to move.

        Returns:
            True if the operation was successfully moved, otherwise False.
        """
        old_key = f"{self.prefix}{operation.op_id}"
        new_key = f"{to_queue_prefix}{operation.op_id}"

        data = operation.to_string()
        value_escaped = data.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        txn = f"""\
        value("{self.lock.lock_key}") = "{self.lock.owner}"
        version("{old_key}") != "0"

        put "{new_key}" "{value_escaped}"
        del "{old_key}"


        """
        return EtcdCtl.txn(txn)

    def watch(self) -> None:
        """Block until at least one operation exists in the queue.

        This method periodically polls the queue prefix and returns once
        an operation is detected
        """
        while True:
            if EtcdCtl.get_first_key_value(self.prefix):
                return
            time.sleep(30)

    def dequeue(self) -> bool:
        """Remove the first operation from the queue.

        The removal is performed using an etcd transaction that ensures
        the lock owner still holds the lock and the operation exists.

        Returns:
            True if the operation was removed successfully, otherwise False.
        """
        kv = EtcdCtl.get_first_key_value(self.prefix)
        if not kv:
            return False
        key, _ = kv

        txn = f"""\
        value("{self.lock.lock_key}") = "{self.lock.owner}"
        version("{key}") != "0"

        del "{key}"


        """
        return EtcdCtl.txn(txn)

    def enqueue(self, operation: Operation) -> bool:
        """Insert a new operation into the queue.

        The method avoids inserting duplicate operations by comparing
        the new operation with the last operation currently in the queue.

        Args:
            operation: Operation to insert.

        Returns:
            True if the operation was inserted, or False if it was skipped
            because it duplicates the most recent operation.
        """
        old_operation = self._peek_last()

        if old_operation is not None and operation == old_operation:
            return False

        op_str = operation.to_string()
        key = f"{self.prefix}{operation.op_id}"
        EtcdCtl.run(["put", key, op_str])
        return True


class RollingOpsManagerV2(Object):
    """Emitters and handlers for rolling ops."""

    def __init__(
        self,
        charm: CharmBase,
        peer_relation_name: str,
        etcd_relation_name: str,
        callback_targets: dict[str, Any],
    ):
        """Register our custom events.

        params:
            charm: the charm we are attaching this to.
            peer_relation_name: peer relation used for rolling ops.
            etcd_relation_name: the relation to integrate with etcd.
            callback_targets: mapping from callback_id -> callable.
        """
        super().__init__(charm, "rolling-ops-manager")
        self._charm = charm
        self.peer_relation_name = peer_relation_name
        self.etcd_relation_name = etcd_relation_name
        self.callback_targets = callback_targets
        self.charm_dir = charm.charm_dir
        self.worker = RollingOpsAsyncWorker(charm, relation_name=peer_relation_name)

        cert = self._get_client_certificate_from_peer()
        mtls_cert = cert[0] if cert else None

        self.etcd = EtcdRequires(
            charm,
            relation_name=etcd_relation_name,
            prefix="/rollingops/",
            mtls_cert=mtls_cert,
        )

        owner = f"{self.model.name}-{self.model.unit.name}".replace("/", "-")

        self.keys = Keys.for_owner(owner)
        self.lock = EtcdLock(self.keys.lock_key, owner)
        self.pending_queue = EtcdOperationQueue(self.keys.pending, self.lock)
        self.inprogress_queue = EtcdOperationQueue(self.keys.inprogress, self.lock)

        charm.on.define_event("rollingop_lock_granted", RollingOpsLockGrantedEvent)

        self.framework.observe(
            charm.on[self.peer_relation_name].relation_departed, self._on_relation_departed
        )
        self.framework.observe(
            charm.on[self.etcd_relation_name].relation_departed, self._on_relation_departed
        )
        self.framework.observe(charm.on.rollingop_lock_granted, self._on_rollingop_granted)
        self.framework.observe(charm.on.update_status, self._on_rollingop_granted)
        self.framework.observe(charm.on.install, self._on_install)
        self.framework.observe(self.etcd.on.etcd_ready, self._on_etcd_ready)
        self.framework.observe(charm.on.leader_elected, self._on_leader_elected)
        self.framework.observe(
            charm.on[self.peer_relation_name].relation_changed, self._on_peer_relation_changed
        )
        self.framework.observe(charm.on.secret_changed, self._on_secret_changed)

    @property
    def _peer_relation(self) -> Relation | None:
        return self.model.get_relation(self.peer_relation_name)
    
    @property
    def _etcd_relation(self) -> Relation | None:
        return self.model.get_relation(self.etcd_relation_name)

    def _on_install(self, event) -> None:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "etcd-client"], check=True)

    def _on_leader_elected(self, event) -> None:
        self._create_and_share_certificate()

    def _on_etcd_ready(self, event) -> None:
        """Configure etcd client access when the etcd relation becomes available.

        It retrieves the endpoints and TLS configuration from the relation databags
        and generates an environment file used by etcdctl commands.
        """
        relation = self._etcd_relation
        if not relation:
            return

        if not self._sync_client_certificate():
            logger.warning("Shared rollingops client certificate is not available yet")
            event.defer()
            return

        endpoints = self.etcd.fetch_relation_field(relation.id, "endpoints")
        tls_ca = self.etcd.fetch_relation_field(relation.id, "tls-ca")

        if not endpoints:
            logger.warning("No etcd endpoints yet")
            return

        client_cert_path, client_key_path = CertificatesManager.client_paths()

        EtcdCtl.write_env_file(
            endpoints=endpoints,
            tls_ca_pem=tls_ca or "",
            client_cert_path=client_cert_path,
            client_key_path=client_key_path,
        )

    def _on_secret_changed(self, event):
        # if event.secret.label == "rollingops-client-cert":
        #    self._sync_client_certificate()
        self._sync_client_certificate()

    def _on_peer_relation_changed(self, event) -> None:
        """React to peer relation changes.

        The leader ensures the shared certificate exists.
        All units try to persist the shared certificate locally if available.
        """
        self._create_and_share_certificate()
        self._sync_client_certificate()

    def _create_and_share_certificate(self) -> None:
        """Ensure the application client certificate exists.

        Only the leader generates the certificate and writes it to the peer
        relation application databag.
        """
        relation = self._peer_relation
        if relation is None or not self.model.unit.is_leader():
            return

        app_data = relation.data[self.model.app]
        secret_id = app_data.get(SECRET_FIELD)

        if secret_id:
            return

        common_name = f"rollingops-{self.model.name}-{self.model.app.name}"
        CertificatesManager.generate(common_name)
        cert_pem, key_pem = CertificatesManager.load_client_cert_and_key()

        secret = self.model.app.add_secret(
            {"cert": cert_pem, "key": key_pem},
        )
        app_data[SECRET_FIELD] = secret.id

    def _get_client_certificate_from_peer(self) -> tuple[str, str] | None:
        """Return the client certificate and key from peer app data.

        Returns:
            A tuple of (certificate_pem, key_pem), or None if not yet available.
        """
        relation = self._peer_relation
        if relation is None:
            return None

        secret_id = relation.data[self.model.app].get(SECRET_FIELD)
        if not secret_id:
            return None

        secret = self.model.get_secret(id=secret_id)
        content = secret.get_content(refresh=True)

        return content["cert"], content["key"]

    def _sync_client_certificate(self) -> bool:
        """Persist the shared client certificate locally on this unit.

        Returns:
            True if the shared certificate was available and written locally,
            otherwise False.
        """
        shared = self._get_client_certificate_from_peer()
        if shared is None:
            logger.debug("Shared rollingops client certificate is not available yet")
            return False

        cert_pem, key_pem = shared
        if CertificatesManager.has_client_cert_and_key(cert_pem, key_pem):
            return True

        CertificatesManager.persist_client_cert_and_key(cert_pem, key_pem)
        return True

    def _on_rollingop_granted(self, event: RollingOpsLockGrantedEvent) -> None:
        if not self._peer_relation or not self._etcd_relation:
            return
        try:
            EtcdCtl.ensure_initialized()
        except EtcdNotConfiguredError:
            return
        logger.info("Received a rolling-op lock granted event.")
        self._on_run_with_lock()

    def _on_relation_departed(self, event: RelationDepartedEvent) -> None:
        """Leader cleanup: if a departing unit was granted, clear the grant.

        This prevents deadlocks when the granted unit leaves the relation.
        """
        unit = event.departing_unit
        if unit == self.model.unit:
            self.worker.stop()
            self.lock.release()

    def request_async_lock(
        self,
        callback_id: str,
        kwargs: dict[str, Any] | None = None,
        max_retry: int | None = None,
    ) -> None:
        """Queue a rolling operation and trigger asynchronous lock acquisition.

        This method creates a new operation representing a callback to execute
        once the distributed lock is granted. The operation is appended to the
        unit's pending operation queue stored in etcd.

        If the operation is successfully enqueued, the background worker process
        responsible for acquiring the distributed lock and processing operations
        is started.

        Args:
            callback_id: Identifier of the registered callback to execute when
                the lock is granted.
            kwargs: Optional keyword arguments passed to the callback when
                executed. Must be JSON-serializable.
            max_retry: Maximum number of retries for the operation.
                - None: retry indefinitely
                - 0: do not retry on failure

        Raises:
            ValueError: If the callback_id is not registered or invalid parameters
            LockNoRelationError: if the etcd relation does not exist
            EtcdNotConfiguredError: if etcd client has not been configured yet
        """
        if callback_id not in self.callback_targets:
            raise ValueError(f"Unknown callback_id: {callback_id}")

        etcd_relation = self.model.get_relation(self.etcd_relation_name)
        if not etcd_relation:
            raise LockNoRelationError

        EtcdCtl.ensure_initialized()

        operation = Operation.create(callback_id, kwargs, max_retry)
        res = self.pending_queue.enqueue(operation)

        if res:
            self.worker.start()
        else:
            logger.info(f"Operation {operation.callback_id} already exists in the queue.")

    def request_sync_lock(self, timeout: int) -> EtcdLease | None:
        """Try to acquire the lock until timeout expires.

        Args:
            timeout: Maximum time in seconds to wait for the lock.

        Returns:
            The granted lease if the lock was acquired, otherwise None.
        """
        lease = EtcdLease()
        lease.grant(60)

        @retry(stop=stop_after_delay(timeout), wait=wait_fixed(30), reraise=True)
        def acquire():
            if not self.lock.try_acquire(lease.id):
                raise RuntimeError("Lock not acquired.")

        try:
            acquire()
            return lease
        except RuntimeError:
            lease.revoke()
            return None

    def release_sync_lock(self, lease: EtcdLease) -> None:
        """Release the lock and revoke the associated lease."""
        self.lock.release()
        lease.revoke()

    def _on_run_with_lock(self) -> None:
        """Execute the current operation while holding the distributed lock.

        This method is triggered when the worker determines that the current
        unit owns the distributed lock. The method retrieves the head operation
        from the in-progress queue and executes its registered callback.

        After execution, the operation is moved to the completed queue and its
        updated state is persisted.
        """
        if not self.lock.is_held():
            logger.debug("Lock is not granted. Operation will not run.")
            return

        operation = self.inprogress_queue.peek()
        if not operation:
            logger.debug("There is no operation to run.")
            return

        callback = self.callback_targets.get(operation.callback_id, "")
        logger.debug(
            "Executing callback_id=%s, attempt=%s", operation.callback_id, operation.attempt
        )

        try:
            result = callback(**operation.kwargs)
        except Exception as e:
            logger.error("Operation failed: %s: %s", operation.callback_id, e)
            result = OperationResult.RETRY_RELEASE

        if result == OperationResult.RETRY_HOLD:
            logger.info(
                "Finished %s. Operation will be retried immediately.", operation.callback_id
            )
            operation.retry_hold()

        elif result == OperationResult.RETRY_RELEASE:
            logger.info("Finished %s. Operation will be retried later.", operation.callback_id)
            operation.retry_release()

        else:
            logger.info("Finished %s. Lock will be released.", operation.callback_id)
            operation.complete()

        moved = self.inprogress_queue.move_operation(self.keys.completed, operation)
        logger.info(f"moved {moved}")


class RollingOpsAsyncWorker(Object):
    """Spawns and manages the external rolling-ops worker process."""

    def __init__(self, charm: CharmBase, relation_name: str):
        super().__init__(charm, "rollingops-async-worker")
        self._charm = charm
        self._peers_name = relation_name
        self._run_cmd = (
            "/usr/bin/juju-exec" if self.model.juju_version.major > 2 else "/usr/bin/juju-run"
        )
        self.owner = f"{self.model.name}-{self.model.unit.name}".replace("/", "-")

    @property
    def _relation(self):
        return self._charm.model.get_relation(self._peers_name)

    @property
    def _unit_data(self):
        return self._relation.data[self.model.unit]

    def start(self) -> None:
        """Start a new worker process."""
        if self._relation is None:
            return

        pid_str = self._unit_data.get("rollingops-worker-pid", "")
        if pid_str:
            try:
                pid = int(pid_str)
            except ValueError:
                pid = -1

            if self._is_pid_alive(pid):
                logger.info(
                    "RollingOps worker already running with PID %s; not starting a new one.", pid
                )
                return

        # Remove JUJU_CONTEXT_ID so juju-run works from the spawned process
        new_env = os.environ.copy()
        new_env.pop("JUJU_CONTEXT_ID", None)

        for loc in new_env.get("PYTHONPATH", "").split(":"):
            path = Path(loc)
            venv_path = (
                path
                / ".."
                / "venv"
                / "lib"
                / f"python{version_info.major}.{version_info.minor}"
                / "site-packages"
            )
            if path.stem == "lib":
                new_env["PYTHONPATH"] = f"{venv_path.resolve()}:{new_env['PYTHONPATH']}"
                break

        worker = self._charm.charm_dir / "lib/charms/rolling_ops/v2" / "rollingops.py"

        pid = subprocess.Popen(
            [
                "/usr/bin/python3",
                "-u",
                str(worker),
                "--run-cmd",
                self._run_cmd,
                "--unit-name",
                self._charm.model.unit.name,
                "--charm-dir",
                str(self._charm.charm_dir),
                "--owner",
                self.owner,
            ],
            cwd=str(self._charm.charm_dir),
            stdout=open("/var/log/rollingops_worker.log", "a"),
            stderr=open("/var/log/rollingops_worker.err", "a"),
            env=new_env,
        ).pid

        self._unit_data.update({"rollingops-worker-pid": str(pid)})
        logger.info("Started RollingOps worker process with PID %s", pid)

    def _is_pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def stop(self) -> None:
        """Stop the running worker process if it exists."""
        if self._relation is None:
            return
        pid_str = self._unit_data.get("rollingops-worker-pid", "")
        if not pid_str:
            return

        pid = int(pid_str)
        try:
            os.kill(pid, signal.SIGINT)
            logger.info("Stopped RollingOps worker process PID %s", pid)
        except OSError:
            pass
        self._unit_data.update({"rollingops-worker-pid": ""})


def main():
    """Juju hook event dispatcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-cmd", required=True)
    parser.add_argument("--unit-name", required=True)
    parser.add_argument("--charm-dir", required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()

    time.sleep(10)

    keys = Keys.for_owner(args.owner)
    lock_lease_ttl = 60
    acquire_retry_sleep = 30
    lock = EtcdLock(keys.lock_key, args.owner)
    pending_queue = EtcdOperationQueue(keys.pending, lock)
    completed_queue = EtcdOperationQueue(keys.completed, lock)
    lease = EtcdLease()

    while True:
        if not pending_queue.peek():
            time.sleep(acquire_retry_sleep)
            continue

        if not lock.is_held():
            if lease.id is None:
                lease.grant(lock_lease_ttl)

            if lock.try_acquire(lease.id):
                print("Lock granted")

            else:
                time.sleep(acquire_retry_sleep)
                continue

        moved = pending_queue.move_head(keys.inprogress)
        if moved:
            # dispatch hook
            print("dispatch hook")
            dispatch_sub_cmd = (
                f"JUJU_DISPATCH_PATH=hooks/rollingop_lock_granted {args.charm_dir}/dispatch"
            )
            res = subprocess.run([args.run_cmd, "-u", args.unit_name, dispatch_sub_cmd])
            res.check_returncode()
        else:
            time.sleep(acquire_retry_sleep)
            continue

        completed_queue.watch()
        operation = completed_queue.peek()
        if operation.result == OperationResult.RETRY_HOLD.value:
            completed_queue.move_head(keys.pending)
            continue

        elif operation.result == OperationResult.RETRY_RELEASE.value:
            completed_queue.move_head(keys.pending)

        else:
            print(completed_queue.dequeue())

        lease.revoke()
        lock.release()
        if not pending_queue.peek():
            break
        time.sleep(acquire_retry_sleep)


if __name__ == "__main__":
    main()
