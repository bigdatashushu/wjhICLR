#!/usr/bin/env python3
"""Start the existing detector service with a verified, local-only BERT encoder."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill3d.segmentation.groundingdino_startup import (
    BERT_REVISION,
    DetectorStartupError,
    bind_local_bert,
    load_local_bert,
    resolve_bert_snapshot,
    snapshot_identity,
)
from skill3d.infra.m5_identity import detector_startup_identity, process_identity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", default=os.environ.get(
        "SKILL3D_DETECTOR_ROOT", "/nas/wangjh/Skill-3D"))
    parser.add_argument("--bert-path", default=os.environ.get("SKILL3D_BERT_PATH"))
    parser.add_argument("--bert-revision", default=BERT_REVISION)
    parser.add_argument("--checkpoint-path", default="checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--port", type=int, default=20024)
    parser.add_argument("--log-dir", default=os.environ.get(
        "SKILL3D_LOG_DIR", "/tmp/skill3d-groundingdino"))
    parser.add_argument("--check-only", action="store_true",
                        help="load the local tokenizer and BERT weights without starting a server")
    args = parser.parse_args(argv)
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("--gpu must be non-negative and --port must be in 1..65535")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = log_dir / "groundingdino_startup.json"
    receipt = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.executable, "gpu": args.gpu, "port": args.port,
        "offline": True, "check_only": args.check_only, "status": "preparing",
    }
    try:
        receipt["process"] = process_identity(os.getpid())
    except OSError:
        receipt["process"] = None

    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        # Set these before importing transformers, torch, or the external service.
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            CUDA_VISIBLE_DEVICES=str(args.gpu), SKILL3D_LOG_DIR=str(log_dir))
        snapshot = resolve_bert_snapshot(args.bert_path, revision=args.bert_revision)
        receipt["bert"] = snapshot_identity(snapshot)
        tokenizer, encoder = load_local_bert(snapshot)
        receipt["bert_loaded"] = True
        if args.check_only:
            receipt.update(status="checked", returncode=0)
            save()
            print(json.dumps(receipt, indent=2), flush=True)
            return 0
        service_root = Path(args.service_root).expanduser().resolve()
        service = service_root / "skill3d/external_experts/GroundingDINO/grounding_dino_server.py"
        if not service.is_file():
            raise DetectorStartupError(f"External GroundingDINO service not found: {service}")
        checkpoint = Path(args.checkpoint_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = service_root / checkpoint
        if not checkpoint.is_file():
            raise DetectorStartupError(f"GroundingDINO checkpoint not found: {checkpoint}")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        import torch
        if not torch.cuda.is_available():
            raise DetectorStartupError(f"CUDA is unavailable on the requested GPU {args.gpu}")
        # The service belongs to a separate repository and uses its own skill3d
        # package. Release ours before importing its absolute package references.
        for name in list(sys.modules):
            if name == "skill3d" or name.startswith("skill3d."):
                del sys.modules[name]
        sys.path.insert(0, str(service_root / "third_party/GroundingDINO"))
        sys.path.insert(0, str(service_root))
        import groundingdino.util.get_tokenlizer as loaders
        bind_local_bert(loaders, snapshot, tokenizer, encoder)
        import groundingdino.util.inference as inference
        import groundingdino
        import transformers
        model_config = service_root / (
            "skill3d/external_experts/GroundingDINO/configs/GroundingDINO_SwinB_cfg.py")
        generation, identity_files = detector_startup_identity(
            service_root=service_root, checkpoint=checkpoint, model_config=model_config,
            bert_snapshot=snapshot,
            groundingdino_root=Path(groundingdino.__file__).parent,
            transformers_root=Path(transformers.__file__).parent,
            launcher=Path(__file__).resolve())
        receipt.update(generation_identity=generation, identity_files=identity_files,
                       service_root=str(service_root))
        original_load_model = inference.load_model

        def logged_load_model(*model_args, **model_kwargs):
            try:
                if (Path(model_args[0]).resolve() != model_config.resolve()
                        or Path(model_args[1]).resolve() != checkpoint.resolve()):
                    raise DetectorStartupError("Actual detector load differs from startup identity")
                return original_load_model(*model_args, **model_kwargs)
            except Exception:
                # The external server catches this exception and exits with 1;
                # retain the original stack rather than only its SystemExit.
                receipt["model_load_traceback"] = traceback.format_exc()
                save()
                traceback.print_exc()
                raise

        inference.load_model = logged_load_model
        os.chdir(service_root)
        sys.argv = [str(service), "--checkpoint_path", str(checkpoint), "--port", str(args.port)]
        receipt.update(status="starting_service", service_root=str(service_root),
                       checkpoint=str(checkpoint), cuda_device=torch.cuda.get_device_name(0))
        save()
        runpy.run_path(str(service), run_name="__main__")
        receipt.update(status="stopped", returncode=0)
        save()
        return 0
    except (Exception, SystemExit) as exc:
        code = exc.code if isinstance(exc, SystemExit) else 1
        if code is None:
            code = 0
        if code == 0:
            receipt.update(status="stopped", returncode=0)
        else:
            receipt.update(status="failed", returncode=1,
                           error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            traceback.print_exc()
        save()
        return 0 if code == 0 else 1
    finally:
        receipt["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    raise SystemExit(main())
