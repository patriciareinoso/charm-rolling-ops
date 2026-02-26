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
#
# Learn more about testing at: https://juju.is/docs/sdk/testing

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Optional

import pytest
from juju.action import Action
from juju.application import Application
from juju.model import Model
from juju.unit import Unit
from pytest_operator.plugin import OpsTest

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%fZ"


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse timestamp string. Return 'now' on errors to avoid selecting invalid timestamps."""
    try:
        dt = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def get_executed_at(unit: Unit, model_name: str) -> str:
    show_unit_json = subprocess.check_output(
        f"JUJU_MODEL={model_name} juju show-unit {unit.name} --format json",
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    show_unit_dict = json.loads(show_unit_json)
    executed_at = show_unit_dict[unit.name]["relation-info"][0]["local-unit"]["data"][
        "executed_at"
    ]

    return _parse_timestamp(executed_at)


@pytest.mark.abort_on_fail
@pytest.mark.group(1)
async def test_smoke(ops_test: OpsTest):
    """Basic smoke test following the default callback implementation.

    Verify that we can deploy, and seem to be able to run a rolling op.
    """
    assert ops_test.model
    assert ops_test.model_full_name
    model: Model = ops_test.model
    model_full_name: str = ops_test.model_full_name

    charm = await ops_test.build_charm(".")
    await asyncio.gather(ops_test.model.deploy(charm, application_name="rolling-ops", num_units=3))

    assert model.applications["rolling-ops"]
    app: Application = model.applications["rolling-ops"]

    await ops_test.model.block_until(lambda: app.status in ("error", "blocked", "active"))
    assert app.status == "active"

    action_type = "restart"
    # Run the restart, with a delay to alleviate timing issues.
    for unit in app.units:
        logger.info(f"Running {action_type} - {unit.name}")
        action: Action = await unit.run_action(action_type, delay=10)
        await action.wait()
        assert (action.results.get("return-code", None) == 0) or (
            action.results.get("Code", None) == "0"
        )

        await model.block_until(lambda: app.status in ("maintenance", "error"), timeout=60)
        assert app.status != "error"

        await model.wait_for_idle(status="active", timeout=600)
        assert get_executed_at()

    for unit in app.units:
        executed_at = get_executed_at(unit=unit, model_name=model_full_name)
        assert executed_at
