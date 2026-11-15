"""Reuse authenticated durable RPC with native Responses payloads."""
from skill3d.cluster.perception_client import RemoteBailianClient
from skill3d.synthesis.xcode_client import XcodeResponsesClient, XcodeMessagesClient, DeepSeekChatClient


class RemoteXcodeClient(RemoteBailianClient, XcodeResponsesClient):
    def __init__(self, *, reasoning_effort="", **kwargs):
        super().__init__(**kwargs)
        self.reasoning_effort = reasoning_effort


class RemoteXcodeMessagesClient(RemoteBailianClient, XcodeMessagesClient):
    def __init__(self, *, reasoning_effort="", **kwargs):
        super().__init__(**kwargs)
        self.reasoning_effort = reasoning_effort


class RemoteDeepSeekChatClient(RemoteBailianClient, DeepSeekChatClient):
    def __init__(self, *, reasoning_effort="", **kwargs):
        super().__init__(**kwargs)
        if reasoning_effort not in self.supported_efforts:
            raise ValueError("unsupported native reasoning effort")
        self.reasoning_effort = reasoning_effort
