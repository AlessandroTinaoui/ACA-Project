#!/usr/bin/env python3
"""gem5 stdlib configuration for the KAN RISC-V SE demo with caches."""

import argparse
from pathlib import Path

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the mini KAN RISC-V binary in gem5 SE mode with caches."
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
    parser.add_argument(
        "--l1i-size",
        default="32KiB",
        help="L1 instruction cache size.",
    )
    parser.add_argument(
        "--l1d-size",
        default="32KiB",
        help="L1 data cache size.",
    )
    parser.add_argument(
        "--l2-size",
        default="256KiB",
        help="Private L2 cache size per core.",
    )
    return parser.parse_args()


args = parse_args()
binary_path = Path(args.binary).resolve()

if not binary_path.exists():
    raise FileNotFoundError(f"RISC-V binary not found: {binary_path}")

requires(isa_required=ISA.RISCV)

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1i_size=args.l1i_size,
    l1d_size=args.l1d_size,
    l2_size=args.l2_size,
)

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
