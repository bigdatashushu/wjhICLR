"""Read-only initial request audit for the frozen paired run."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/cv3d-simple-paired-20261101-r02")


def audit(root=ROOT, phase="test"):
    root = Path(root)
    revised_source = root / "ops/concurrency8/source"
    source = revised_source if revised_source.is_dir() else root / "source_frozen"
    sys.path[:0] = [str(source / "src"), str(source / "tests")]
    from CV3D.common import InferenceConfig, digest, read_json
    from CV3D.client import NativeToolClient
    from CV3D.data import index_prepared, replay_inputs
    from CV3D.inference import prepare_request
    from CV3D.replay import replay_prepared
    from skill3d.synthesis.request_audit import payload_bytes
    manifest = read_json(root / "manifest.json")
    config = manifest["config"]
    public = {p.problem.sample_id: p for p in replay_inputs(
        config["public_manifest"], config["dataset_root"], config["official_annotations"])}
    index = index_prepared(config["prepared_root"])
    client = NativeToolClient(InferenceConfig(timeout_s=config["timeout_s"]))
    out = {"status": "no_completed_pair", "phase": phase, "inspection_model_requests": 0}
    for key in manifest["test_ids"] if phase == "test" else manifest["train_protocol_ids"]:
        directories = {
            arm: root / phase / "episodes" / arm / key for arm in ("tools", "direct")}
        if not all((p / "result.json").is_file() for p in directories.values()):
            continue
        records = {}
        for arm, directory in directories.items():
            p = public[key]
            *_, identity = prepare_request(p, index[p.problem.input_sha256], arm=arm)
            assert identity == manifest["request_identities"][arm][key]
            prepared, messages, tools, png, identity = prepare_request(
                p, index[p.problem.input_sha256], arm=arm)
            payload = client.payload(messages, tools)
            intent = read_json(directory / "0000.intent.json")
            assert intent["wire_payload_sha256"] == hashlib.sha256(payload_bytes(payload)).hexdigest()
            response = read_json(directory / "0000.response.json")
            assert response["model"] == "qwen3.8-flash"
            assert intent["model_configuration"] == client.config.public_dict()
            image_hashes = [
                hashlib.sha256(__import__("base64").b64decode(
                    part["image_url"]["url"].split(",", 1)[1])).hexdigest()
                for part in messages[0]["content"] if part["type"] == "image_url"
            ]
            assert len(image_hashes) == (2 if arm == "tools" else 1)
            if arm == "direct":
                assert "tools" not in payload and "tool_choice" not in payload
                assert messages[0]["content"][0]["text"] == p.official_prompt
            records[arm] = {
                "wire_payload_sha256": intent["wire_payload_sha256"],
                "image_hashes": image_hashes, "model": response["model"],
                "status": read_json(directory / "result.json")["status"],
            }
            restored = replay_prepared(
                p, directory, prepared=prepared, messages=messages, tools=tools,
                png=png, identity=identity, arm=arm, client=client,
                expected_identity=manifest["request_identities"][arm][key])
            records[arm].update(
                verified_requests=restored["verified_requests"],
                delivered_tool_observations=restored["attempted_tool_calls"])
        assert records["tools"]["image_hashes"][0] == records["direct"]["image_hashes"][0]
        out.update(status="passed", sample_id=key, arms=records,
                   original_image_bytes_equal=True, first_request_identity_verified=True,
                   model_parameters_equal=True, test_inputs_not_modified=True)
        return out
    return out


if __name__ == "__main__":
    print(json.dumps(audit(phase=sys.argv[1] if len(sys.argv) > 1 else "test"),
                     ensure_ascii=False))
