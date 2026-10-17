#!/usr/bin/env python3
"""Run actual M5 against frozen frames/geometry with the optional local reviewer.

Requires a running unchanged GroundingDINO service and existing SAM2 checkpoint.
A historical *parser-only* receipt can be reused explicitly when the Flash key is absent.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.online.runner import _episode_frame_provenance
from skill3d.schemas.episode import VSIBenchEpisode
from skill3d.schemas.reconstruction import ReconstructionArtifact
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.segmentation.intern_decision_client import InternDecisionClient
from skill3d.segmentation.sam2_tracker import bind_objects_for_scene
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_context import current_request_phase


class HistoricalParser:
    model = 'qwen3.8-flash-historical-parser-only'
    max_output_tokens = 2048
    last_usage = {}

    def __init__(self, path):
        self.path = path
        data = json.loads(path.read_text())
        self.receipt = data['stats']['object_parse']
        if self.receipt['status'] != 'parsed':
            raise ValueError('Historical parser was not successful')
        if self.receipt['identity']['model']['model'] != 'qwen3.8-flash':
            raise ValueError('Historical parser was not the authorized Flash model')

    def audit_identity(self, *, seed=None):
        return {'provider': 'historical-parser-only', 'source': file_identity(self.path),
                'seed': seed, 'cache_eligible': False, 'new_api_request': False}

    def chat(self, messages, **_kwargs):
        if current_request_phase() != 'm5_object_parse':
            raise ValueError('Historical parser cannot judge vision or relations')
        if json.loads(messages[-1]['content']) != self.receipt['identity']['input']:
            raise ValueError('Historical parser input differs')
        return next(r['raw_response'] for r in self.receipt['requests'] if 'raw_response' in r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', default='http://127.0.0.1:18445')
    p.add_argument('--identity', default='')
    p.add_argument('--relation', choices=['qwen', 'frozen-rules'], default='qwen')
    p.add_argument('--historical-parser-result', type=Path)
    p.add_argument('--binary-operating-point', type=Path)
    p.add_argument('--frozen-quality-manifest', type=Path)
    args = p.parse_args()
    episode = VSIBenchEpisode.model_validate_json((args.frozen / 'episode.json').read_text())
    artifact = ReconstructionArtifact.model_validate_json((args.frozen / 'artifact.json').read_text())
    frames = [np.load(path, allow_pickle=False) for path in sorted(args.frozen.glob('frame_*.npy'))]
    if len(frames) != 32:
        raise ValueError('Require all 32 frozen frames')
    cloud = BailianClient()
    parser = HistoricalParser(args.historical_parser_result) if args.historical_parser_result else cloud
    if not args.binary_operating_point and args.relation == 'qwen' and not cloud.configured:
        raise ValueError('DASHSCOPE_API_KEY absent for requested Qwen relation')
    if not args.historical_parser_result and not cloud.configured:
        raise ValueError('DASHSCOPE_API_KEY absent; supply explicit historical parser receipt')
    # Historical parser cannot serve the Qwen relation stage.
    if not args.binary_operating_point and args.historical_parser_result and args.relation != 'frozen-rules':
        raise ValueError('Use live parser for Qwen relation or explicit frozen-rules experiment')
    identity = InternDecisionClient(args.endpoint, expected_identity_sha256=args.identity).identity()
    config = {'provider': 'intern-decision', 'endpoint': args.endpoint, 'timeout_s': 60,
              'expected_identity_sha256': content_sha256(identity), 'relation': args.relation}
    if args.binary_operating_point:
        from skill3d.segmentation.intern_binary_review import PROTOCOL_SHA256
        frozen = json.loads(args.binary_operating_point.read_text())
        if (frozen.get('status') != 'frozen' or frozen.get('protocol_sha256') != PROTOCOL_SHA256
                or frozen.get('model_identity_sha256') != content_sha256(identity)
                or not args.frozen_quality_manifest):
            raise ValueError('binary operating point or quality identity unavailable')
        config.update(mode='binary', prompt_version=frozen['prompt_version'],
            yes_threshold=frozen['threshold'], quality_manifest=str(args.frozen_quality_manifest),
            quality_manifest_sha256=file_identity(args.frozen_quality_manifest)['sha256'])
        config.pop('relation')
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = {name: np.load(getattr(artifact, name), allow_pickle=False) for name in
              ['depth_maps', 'c2w_list', 'intrinsics', 'depth_conf', 'point_map']}
    objects, stats, notes = bind_objects_for_scene(frames, None, episode.question,
        depth_maps=arrays['depth_maps'], c2w_list=arrays['c2w_list'], intrinsics=arrays['intrinsics'],
        out_dir=args.output, scene_name=episode.scene_name, point_conf=arrays['depth_conf'],
        depth_conf=arrays['depth_conf'], vlm_client=parser, seed=137,
        frame_set_hash=episode.frame_set.frame_set_hash, question_type=episode.question_type,
        options=episode.options, qa_id=episode.qa_id,
        scene_identity=artifact.reconstruction_input_identity or artifact.frame_set_hash or artifact.artifact_id,
        frame_provenance=_episode_frame_provenance(episode, frames), point_map=arrays['point_map'],
        geometry_usable=bool(artifact.quality and artifact.quality.main_gate_passed),
        semantic_review_config=config)
    verdicts = [validate_binding(o, depth_maps=arrays['depth_maps'], n_frames=32,
                generation_sha256=stats['generation']['sha256'], check_receipt=True).model_dump(mode='json')
                for o in objects]
    result = {'stats': stats, 'objects': [o.model_dump(mode='json') for o in objects],
              'validation': verdicts, 'notes': notes,
              'parser_new_api_request': args.historical_parser_result is None,
              'parser_receipt': str(args.historical_parser_result or '')}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'objects': len(objects), 'validated': sum(v['status'] == 'validated' for v in verdicts),
                      'status': stats['status'], 'elapsed_s': stats['elapsed_s']}))

if __name__ == '__main__':
    main()
