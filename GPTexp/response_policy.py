"""Provider metadata is optional under the explicitly selected correlation policy."""
STRICT = "strict"
CORRELATION = "request_context"


def policy(config):
    value = config.get("response_metadata_policy", STRICT)
    if value not in (STRICT, CORRELATION):
        raise ValueError("unknown_response_metadata_policy")
    return value


def continuation_config(config, budget):
    from copy import deepcopy

    result = deepcopy(config)
    result["max_requests"] = budget
    result["response_metadata_policy"] = CORRELATION
    result["upstream_policy"].update(
        require_reported_sampling_match=False,
        require_reported_instructions_match=False,
        require_reported_output_limit_match=False)
    return result
