"""Command-line choice of refinement passes, shared by the demos.

``--resolution`` runs every rung up to its value, as the release does. ``--ladder``
names the passes instead: ``--ladder 512 1024 1536 2048`` runs all four rungs,
``--ladder 512 1024 2048`` revoxelizes the 1024 mesh straight onto the 2048 grid,
and ``--ladder 2048`` refines the coarse mesh in a single 256 pass.
"""

from __future__ import annotations

import argparse

from sparc3d import Sparc3DPipeline
from sparc3d.pipelines.sparc3d_pipeline import (
    OUTPUT_RESOLUTIONS,
    STREAMED_ONLY_RESOLUTIONS,
)


def add_ladder_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--resolution``/``--ladder`` and the streamed-decoding options."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resolution",
        type=int,
        choices=OUTPUT_RESOLUTIONS,
        help="Run every pass up to this output resolution (default 1024).",
    )
    group.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        choices=OUTPUT_RESOLUTIONS,
        metavar="RES",
        help="Output resolutions of the passes to run, increasing, e.g. "
        "512 1024 2048, or 2048 alone for a single pass from the coarse mesh.",
    )
    parser.add_argument(
        "--stream-devices",
        nargs="*",
        help="Stream decoding in tiles, optionally sharing them with these extra "
        "GPUs. Always on for 2048, whose decode does not fit one device.",
    )


def configure_ladder(
    pipeline: Sparc3DPipeline, args: argparse.Namespace
) -> int | tuple[int, ...]:
    """Prepare ``pipeline`` for the requested passes and return its ``resolution``.

    Raises:
        SystemExit: The ladder is not strictly increasing.
    """
    if args.ladder is not None:
        if any(b <= a for a, b in zip(args.ladder, args.ladder[1:])):
            raise SystemExit("--ladder resolutions must be strictly increasing")
        resolution: int | tuple[int, ...] = tuple(args.ladder)
        target = args.ladder[-1]
    else:
        resolution = target = args.resolution or 1024

    if target in STREAMED_ONLY_RESOLUTIONS or args.stream_devices is not None:
        # Out-of-core decoding has no feature limit, so it runs the full decoder
        # and keeps only the stage in use on the GPU, as 2048 was validated.
        pipeline.enable_streamed_decoding(args.stream_devices or [])
        pipeline.decode_feature_ceiling = None
        pipeline.set_decoder_skip_points(2**62)
        pipeline.offload_idle_models = True
    return resolution
