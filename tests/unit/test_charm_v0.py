# Copyright 2022 Canonical Ltd.
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
#
# Learn more about testing at: https://juju.is/docs/sdk/testing

import time
import unittest
from unittest.mock import Mock

from charms.rolling_ops.v0.rollingops import RollingOpsManager
from ops import CharmBase
from ops.framework import StoredState
from ops.model import ActiveStatus, MaintenanceStatus, WaitingStatus
from ops.testing import Harness

META_V0 = """
name: rolling-ops
peers:
  restart:
    interface: rolling_op
"""
ACTIONS_V0 = """
restart:
    description: Restarts the example service
    params:
        delay:
        description: "Introduce an artificial delay (for testing)."
        type: integer
        default: 0

custom-restart:
    description: Example restart with a custom callback function. Used in testing
    params:
        delay:
        description: "Introduce an artificial delay (for testing)."
        type: integer
        default: 0
"""


class CharmRollingOpsCharmV0(CharmBase):
    """Charm the service."""

    _stored = StoredState()

    def __init__(self, *args):
        super().__init__(*args)

        self.restart_manager = RollingOpsManager(
            charm=self, relation="restart", callback=self._restart
        )

        self.framework.observe(self.on.install, self._on_install)
        self.framework.observe(self.on.restart_action, self._on_restart_action)
        self.framework.observe(self.on.custom_restart_action, self._on_custom_restart_action)

        # Sentinel for testing (omit from production charms)
        self._stored.set_default(restarted=False)
        self._stored.set_default(delay=None)

    def _restart(self, event):
        # In a production charm, we'd perhaps import the systemd library, and run
        # systemd.restart_service.  Here, we just set a sentinel in our stored state, so
        # that we can run our tests.
        if self._stored.delay:
            time.sleep(int(self._stored.delay))
        self._stored.restarted = True

        self.model.get_relation(self.restart_manager.name).data[self.unit].update({
            "restart-type": "restart"
        })

    def _custom_restart(self, event):
        if self._stored.delay:
            time.sleep(int(self._stored.delay))

        self.model.get_relation(self.restart_manager.name).data[self.unit].update({
            "restart-type": "custom-restart"
        })

    def _on_install(self, event):
        self.unit.status = ActiveStatus()

    def _on_restart_action(self, event):
        self._stored.delay = event.params.get("delay")
        self.on[self.restart_manager.name].acquire_lock.emit()

    def _on_custom_restart_action(self, event):
        self._stored.delay = event.params.get("delay")
        self.on[self.restart_manager.name].acquire_lock.emit(callback_override="_custom_restart")


class TestCharm(unittest.TestCase):
    def setUp(self):
        self.harness = Harness(CharmRollingOpsCharmV0, meta=META_V0, actions=ACTIONS_V0)
        self.addCleanup(self.harness.cleanup)
        self.harness.begin_with_initial_hooks()
        # self.harness.begin()

    def test_restart(self):
        # Verify that our _restart handler gets called when we call RunWithLock
        self.harness.charm.on["restart"].run_with_lock.emit()
        self.assertTrue(self.harness.charm._stored.restarted)

    def test_acquire(self):
        # A human operator runs the "restart" action.
        action_event = Mock()
        action_event.callback_override = "_custom_restart"
        self.harness.charm.restart_manager._on_acquire_lock(action_event)

        data = self.harness.charm.model.relations["restart"][0].data

        # The result should be that we set a lock request on our relation data.
        self.assertEqual(data[self.harness.model.unit]["state"], "acquire")

    def test_peers(self):
        # Set unit 0 as the leader.
        # Add a peer relation to a unit 1.
        self.harness.set_leader(True)
        self.harness.add_relation_unit(0, "rolling-ops/1")
        self.harness.update_relation_data(0, "rolling-ops/0", {"state": "acquire"})
        self.harness.update_relation_data(0, "rolling-ops/1", {"state": "acquire"})

        unit_0 = self.harness.charm.model.get_unit("rolling-ops/0")
        unit_1 = self.harness.charm.model.get_unit("rolling-ops/1")
        rel_data = self.harness.charm.model.relations["restart"][0].data

        # Unit 1 should have requested the lock, and been granted the lock.
        self.assertEqual(rel_data[unit_1]["state"], "acquire")
        self.assertEqual(rel_data[self.harness.model.app][str(unit_1)], "granted")

        self.assertEqual(
            self.harness.charm.model.app.status, MaintenanceStatus("Beginning rolling restart")
        )
        self.assertEqual(
            self.harness.charm.model.unit.status, WaitingStatus("Awaiting restart operation")
        )

        # Unit 0 should have requested the lock, but not yet granted the lock to itself.
        self.assertEqual(rel_data[unit_0]["state"], "acquire")

        # Now we simulate unit 1 processing and releasing the lock.
        self.harness.update_relation_data(0, "rolling-ops/1", {"state": "release"})

        # This should result in unit 0 granting itself the lock, and executing.
        # Both units should end up in the "release" and "idle" state.
        rel_data = self.harness.charm.model.relations["restart"][0].data
        self.assertEqual(rel_data[unit_1]["state"], "release")
        self.assertEqual(rel_data[unit_0]["state"], "release")
        self.assertEqual(rel_data[self.harness.model.app][str(unit_1)], "idle")
        self.assertEqual(rel_data[self.harness.model.app][str(unit_0)], "idle")

        self.assertEqual(self.harness.charm.model.app.status, ActiveStatus())
        self.assertEqual(self.harness.charm.model.unit.status, ActiveStatus())
