"""4-stage curriculum learning for facade drone training.

Aligned with ROADMAP.md curriculum design:
  Stage 0: Calm air hover (0-500 epochs)
  Stage 1: Light wind (500-1500 epochs)
  Stage 2: Medium wind + spray (1500-3000 epochs)
  Stage 3: Full randomization (3000+ epochs)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    """Parameters for a single curriculum stage."""

    wind_max: float
    spray_enabled: bool
    spray_force_max: float
    init_pos_noise: float
    description: str


# Stage definitions
STAGES: dict[int, CurriculumStage] = {
    0: CurriculumStage(
        wind_max=0.0,
        spray_enabled=False,
        spray_force_max=0.0,
        init_pos_noise=0.1,
        description="Calm air: hover at d=1.5m from wall",
    ),
    1: CurriculumStage(
        wind_max=3.0,
        spray_enabled=False,
        spray_force_max=0.0,
        init_pos_noise=0.3,
        description="Light wind (0-3 m/s): maintain distance",
    ),
    2: CurriculumStage(
        wind_max=6.0,
        spray_enabled=True,
        spray_force_max=8.0,
        init_pos_noise=0.5,
        description="Medium wind (0-6 m/s) + low-pressure spray",
    ),
    3: CurriculumStage(
        wind_max=12.0,
        spray_enabled=True,
        spray_force_max=15.0,
        init_pos_noise=1.0,
        description="Full randomization: strong wind + high-pressure spray",
    ),
}

# Epoch thresholds for stage transitions
STAGE_THRESHOLDS: list[int] = [0, 500, 1500, 3000]


def get_stage(epoch: int) -> int:
    """Return the curriculum stage index for the given training epoch."""
    stage = 0
    for i, threshold in enumerate(STAGE_THRESHOLDS):
        if epoch >= threshold:
            stage = i
    return stage


def get_stage_params(epoch: int) -> CurriculumStage:
    """Return curriculum parameters for the given training epoch."""
    return STAGES[get_stage(epoch)]
