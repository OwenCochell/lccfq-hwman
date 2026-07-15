"""Transport selection for the QICK board connection.

CQEDToolbox's ``QBoardConfig.generate_soccfg()`` builds its ``(soc, soccfg)``
pair from a Pyro4 nameserver lookup. qcat's ``make_proxy()`` returns the same
pair over a gRPC channel, so overriding that one method swaps the transport and
leaves ``config_()`` -- the parameter-manager logic -- untouched.
"""

import logging
from pathlib import Path
from typing import Any

import grpc
from cqedtoolbox.protocols.configs.qick_config import QickConfig
from qick import QickConfig as QickSocConfig

from hwman.config import HwmanSettings

logger = logging.getLogger(__name__)


def _read(path: Path | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).read_bytes()


def build_channel_credentials(settings: HwmanSettings) -> grpc.ChannelCredentials | None:
    """Build gRPC channel credentials for the qcat connection.

    Returns None for a plaintext channel, which is what make_proxy expects when
    the board's qcat server is running without TLS.
    """
    if settings.qick_grpc_tls_ca is None and settings.qick_grpc_tls_cert is None:
        return None

    return grpc.ssl_channel_credentials(
        root_certificates=_read(settings.qick_grpc_tls_ca),
        private_key=_read(settings.qick_grpc_tls_key),
        certificate_chain=_read(settings.qick_grpc_tls_cert),
    )


class GrpcQickConfig(QickConfig):
    """QickConfig that reaches the board over qcat gRPC instead of Pyro4."""

    def __init__(
        self,
        params: Any,
        host: str,
        port: int = 8000,
        channel_credentials: grpc.ChannelCredentials | None = None,
        **kwargs: Any,
    ) -> None:
        """
        :param params: Proxy instance of the parameter manager, as for QBoardConfig.
        :param host: Address of the qcat server running on the QICK board.
        :param port: Port of the qcat server.
        :param channel_credentials: TLS credentials; None for a plaintext channel.
        """
        super().__init__(params=params, **kwargs)
        self.host = host
        self.port = port
        self.channel_credentials = channel_credentials

    def generate_soccfg(self) -> QickSocConfig:
        """Connect to the qcat server and store the soc/soccfg pair.

        :return: soccfg: QickConfig instance.
        """
        # Imported lazily: qcat_grpc is an optional dependency, so the pyro
        # transport stays usable without it installed.
        from qcat_grpc import make_proxy

        logger.info(f"Connecting to qcat server at {self.host}:{self.port}")
        soc, soccfg = make_proxy(
            self.host,
            self.port,
            channel_credentials=self.channel_credentials,
        )
        self.soc = soc
        self.soccfg = soccfg
        logger.info("Generated soccfg over gRPC")
        logger.info(soccfg)
        return soccfg
