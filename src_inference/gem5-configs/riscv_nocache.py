#!/usr/bin/env python3
"""gem5 stdlib configuration for the first KAN RISC-V SE NoCache demo."""

import argparse
from pathlib import Path

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


# Parser of the script
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the mini KAN RISC-V binary in gem5 SE mode."
    )
    parser.add_argument(
        "--binary",
        default="build/riscv/kan_demo_riscv",
        help="Path to the local statically linked RISC-V binary.",
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=1024,
        help="Number of uniformly spaced inputs passed to the demo binary.",
    )
    return parser.parse_args()


args = parse_args()
binary_path = Path(args.binary).resolve()

if not binary_path.exists():
    raise FileNotFoundError(f"RISC-V binary not found: {binary_path}")

# RISCV requirements
requires(isa_required=ISA.RISCV)

cache_hierarchy = NoCache()

memory = SingleChannelDDR3_1600(size="512MiB")

processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING,
    isa=ISA.RISCV,
    num_cores=1,
)

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

binary = BinaryResource(local_path=str(binary_path), architecture=ISA.RISCV)
board.set_se_binary_workload(binary, arguments=[str(args.num_inputs)])

simulator = Simulator(board=board)
simulator.run()
