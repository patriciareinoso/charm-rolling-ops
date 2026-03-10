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
from pathlib import Path
from unittest.mock import patch

from charms.rolling_ops.v2.rollingops import (
    SECRET_FIELD,
)
from ops.testing import Context, PeerRelation, Secret, State

from charm import CharmRollingOpsCharm


def test_leader_elected_creates_shared_secret_and_stores_id():
    ctx = Context(CharmRollingOpsCharm)

    peer_relation = PeerRelation(endpoint="restart")

    state_in = State(
        leader=True,
        relations={peer_relation},
    )

    with (
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.exists",
            return_value=False,
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.generate",
            return_value=None,
        ) as mock_generate,
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.load_client_cert_and_key",
            return_value=("CERT_PEM", "KEY_PEM"),
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.client_paths",
            return_value=(Path("/tmp/client.pem"), Path("/tmp/client.key")),
        ),
        patch("charms.rolling_ops.v2.rollingops.EtcdCtl"),
    ):
        state_out = ctx.run(ctx.on.leader_elected(), state_in)

        peer_out = next(r for r in state_out.relations if r.endpoint == "restart")
        assert SECRET_FIELD in peer_out.local_app_data
        assert peer_out.local_app_data[SECRET_FIELD].startswith("secret:")
        mock_generate.assert_called_once()


def test_leader_elected_does_not_regenerate_when_secret_already_exists():
    ctx = Context(CharmRollingOpsCharm)
    peer_relation = PeerRelation(
        endpoint="restart", local_app_data={SECRET_FIELD: "secret:existing"}
    )
    secret = Secret(
        id="secret:existing",
        owner="application",
        tracked_content={
            "cert": "CERT_PEM",
            "key": "KEY_PEM",
        },
    )

    state_in = State(leader=True, relations={peer_relation}, secrets=[secret])

    with (
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.exists",
            return_value=False,
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.generate",
            return_value=None,
        ) as mock_generate,
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.load_client_cert_and_key",
            return_value=("CERT_PEM", "KEY_PEM"),
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.client_paths",
            return_value=(Path("/tmp/client.pem"), Path("/tmp/client.key")),
        ),
        patch("charms.rolling_ops.v2.rollingops.EtcdCtl"),
    ):
        state_out = ctx.run(ctx.on.leader_elected(), state_in)

        peer_out = next(r for r in state_out.relations if r.endpoint == "restart")
        assert peer_out.local_app_data[SECRET_FIELD] == "secret:existing"
        mock_generate.assert_not_called()


def test_non_leader_does_not_create_shared_secret():
    ctx = Context(CharmRollingOpsCharm)
    peer_relation = PeerRelation(endpoint="restart")

    state_in = State(
        leader=False,
        relations=[peer_relation],
    )

    with (
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.exists",
            return_value=False,
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.generate",
            return_value=None,
        ) as mock_generate,
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.load_client_cert_and_key",
            return_value=("CERT_PEM", "KEY_PEM"),
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.client_paths",
            return_value=(Path("/tmp/client.pem"), Path("/tmp/client.key")),
        ),
        patch("charms.rolling_ops.v2.rollingops.EtcdCtl"),
    ):
        state_out = ctx.run(ctx.on.relation_changed(peer_relation, remote_unit=1), state_in)

        peer_out = next(r for r in state_out.relations if r.endpoint == "restart")
        assert SECRET_FIELD not in peer_out.local_app_data
        mock_generate.assert_not_called()


def test_relation_changed_syncs_local_certificate_from_secret():
    ctx = Context(CharmRollingOpsCharm)
    peer_relation = PeerRelation(
        endpoint="restart", local_app_data={SECRET_FIELD: "secret:rollingops-cert"}
    )

    secret = Secret(
        id="secret:rollingops-cert",
        tracked_content={"cert": "CERT_PEM", "key": "KEY_PEM"},
    )

    state_in = State(
        leader=False,
        relations=[peer_relation],
        secrets=[secret],
    )

    with (
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.exists",
            return_value=False,
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.generate",
            return_value=None,
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.load_client_cert_and_key",
            return_value=("CERT_PEM", "KEY_PEM"),
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.has_client_cert_and_key",
            return_value=False,
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.client_paths",
            return_value=(Path("/tmp/client.pem"), Path("/tmp/client.key")),
        ),
        patch(
            "charms.rolling_ops.v2.rollingops.CertificatesManager.persist_client_cert_and_key",
            return_value=(Path("/tmp/client.pem"), Path("/tmp/client.key")),
        ) as mock_persit,
        patch("charms.rolling_ops.v2.rollingops.EtcdCtl"),
    ):
        ctx.run(ctx.on.relation_changed(peer_relation, remote_unit=1), state_in)
        mock_persit.assert_called_once_with("CERT_PEM", "KEY_PEM")
