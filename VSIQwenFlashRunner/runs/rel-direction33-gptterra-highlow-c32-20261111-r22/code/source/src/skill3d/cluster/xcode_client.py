"""Reuse authenticated durable RPC with native Responses payloads."""
from skill3d.cluster.perception_client import RemoteBailianClient
from skill3d.synthesis.xcode_client import XcodeResponsesClient


class RemoteXcodeClient(RemoteBailianClient, XcodeResponsesClient):
    def __init__(self, *, reasoning_effort="", **kwargs):
        super().__init__(**kwargs)
        self.reasoning_effort = reasoning_effort
