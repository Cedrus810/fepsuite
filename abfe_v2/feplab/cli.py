"""Command-line entry of the feplab pipeline.

Invoked by ``abfe_v2/pipeline.zsh`` (which controller.zsh sources/executes):

    python3 -m feplab.cli query all        # list the named stages
    python3 -m feplab.cli query <stage>    # controller protocol line
    python3 -m feplab.cli run <stage>      # execute a stage (run mode)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import Config
from .errors import ConfigError, PipelineError
from .phases import STAGES, query_line, stage_index
from .runner import RunContext, Runner
from .stages import Pipeline


def _query_config() -> Config:
    """Config for query resolution (runs on the submit node, before $ID)."""
    extra = os.environ.get("FEPLAB_PARA_CONF")
    base = Path(extra) if extra else Path.cwd() / "para_conf.zsh"
    if not base.exists():
        return Config({}, sources=[])
    return Config.load(Path.cwd(), None, extra_path=extra)


def do_query(stage: str) -> None:
    if stage == "all":
        print(" ".join(s.key for s in STAGES))
        return
    print(query_line(stage, _query_config()))


def do_run(stage: str) -> None:
    stage_index(stage)  # validate the name early
    ctx = RunContext.from_env()
    # Normalize the working directory to the run root, so all the
    # rundir-relative file paths (mdp/, <ID>/...) agree everywhere.
    os.chdir(ctx.rundir)
    config = Config.load(ctx.rundir, ctx.run_id,
                         extra_path=os.environ.get("FEPLAB_PARA_CONF"))
    config.validate()
    runner = Runner(ctx)
    Pipeline(config=config, ctx=ctx, runner=runner).run_stage(stage)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="feplab", description="FEP-ABFE pipeline v2 (see abfe_v2/README.md)")
    sub = parser.add_subparsers(dest="command", required=True)
    parser_query = sub.add_parser("query", help="controller.zsh protocol: list stages")
    parser_query.add_argument("stage", help="stage name or 'all'")
    parser_run = sub.add_parser("run", help="run one stage")
    parser_run.add_argument("stage", help="stage name (see: query all)")
    args = parser.parse_args(argv)

    try:
        if args.command == "query":
            do_query(args.stage)
        else:
            do_run(args.stage)
    except (PipelineError, ConfigError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
