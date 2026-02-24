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
import logging

import pytest
from juju.action import Action
from juju.application import Application
from juju.model import Model
from pytest_operator.plugin import OpsTest

logger = logging.getLogger(__name__)


@pytest.mark.abort_on_fail
@pytest.mark.group(1)
async def test_smoke(ops_test: OpsTest):
    """Basic smoke test following the default callback implementation.

    Verify that we can deploy, and seem to be able to run a rolling op.
    """
    # to spare the typechecker errors
    assert ops_test.model
    assert ops_test.model_full_name
    model: Model = ops_test.model

    # Deploy, and verify deployment
    charm = await ops_test.build_charm(".")
    await asyncio.gather(ops_test.model.deploy(charm, application_name="rolling-ops", num_units=3))

    # to spare the typechecker errors
    assert model.applications["rolling-ops"]
    app: Application = model.applications["rolling-ops"]

    await ops_test.model.block_until(lambda: app.status in ("error", "blocked", "active"))
    assert app.status == "active"

    for action_type in ["restart"]:
        # Run the restart, with a delay to alleviate timing issues.
        for unit in app.units:
            logger.info(f"{action_type} - {unit.name}")
            action: Action = await unit.run_action(action_type, delay=10)
            await action.wait()
            assert (action.results.get("return-code", None) == 0) or (
                action.results.get("Code", None) == "0"
            )

        await model.block_until(lambda: app.status in ("maintenance", "error"), timeout=60)
        assert app.status != "error"

        await model.wait_for_idle(status="active", timeout=600)
