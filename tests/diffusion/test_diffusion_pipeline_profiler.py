# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.profiler import diffusion_pipeline_profiler as profiler_module
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _InlinePipeline(DiffusionPipelineProfilerMixin):
    def __init__(self, *, enabled: bool):
        self.setup_diffusion_pipeline_profiler(profiler_targets=[], enable_diffusion_pipeline_profiler=enabled)

    def forward(self):
        return None


def test_inline_stage_records_synchronized_duration(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(profiler_module.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(profiler_module.current_omni_platform, "synchronize", synchronize)
    pipeline = _InlinePipeline(enabled=True)

    with pipeline.profile_stage("diffuse"):
        pass

    assert pipeline.stage_durations["_InlinePipeline.diffuse"] >= 0.0
    assert synchronize.call_count == 2


def test_disabled_inline_stage_does_not_access_platform_or_record(monkeypatch):
    def fail_if_called():
        raise AssertionError("Disabled profiling accessed the platform")

    monkeypatch.setattr(profiler_module.current_omni_platform, "is_available", fail_if_called)
    monkeypatch.setattr(profiler_module.current_omni_platform, "synchronize", fail_if_called)
    pipeline = _InlinePipeline(enabled=False)

    with pipeline.profile_stage("diffuse"):
        pass

    assert not hasattr(pipeline, "_stage_durations")


def test_inline_stage_records_and_propagates_exception(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(profiler_module.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(profiler_module.current_omni_platform, "synchronize", synchronize)
    pipeline = _InlinePipeline(enabled=True)

    with pytest.raises(RuntimeError, match="stage failed"):
        with pipeline.profile_stage("empty_cache"):
            raise RuntimeError("stage failed")

    assert pipeline.stage_durations["_InlinePipeline.empty_cache"] >= 0.0
    assert synchronize.call_count == 2
