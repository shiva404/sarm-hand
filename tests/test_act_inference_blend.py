"""Tests for ACT inference blend ramp."""

from sarm_hand.policy import _blend_action_with_present, _inference_blend_alpha


def test_blend_holds_present_at_alpha_zero():
    obs = {"shoulder_lift.pos": -97.5, "elbow_flex.pos": 99.0}
    policy = {"shoulder_lift.pos": -42.9, "elbow_flex.pos": 46.0}
    out = _blend_action_with_present(policy, obs, 0.0)
    assert out["shoulder_lift.pos"] == -97.5
    assert out["elbow_flex.pos"] == 99.0


def test_blend_full_policy_at_alpha_one():
    obs = {"shoulder_lift.pos": -97.5}
    policy = {"shoulder_lift.pos": -42.9}
    out = _blend_action_with_present(policy, obs, 1.0)
    assert out["shoulder_lift.pos"] == -42.9


def test_blend_midpoint():
    obs = {"shoulder_lift.pos": -100.0}
    policy = {"shoulder_lift.pos": -40.0}
    out = _blend_action_with_present(policy, obs, 0.5)
    assert out["shoulder_lift.pos"] == -70.0


def test_inference_blend_skips_replan_when_temporal_ensemble():
    assert _inference_blend_alpha(
        100,
        inference_blend_steps=0,
        replan_blend_steps=8,
        n_action_steps=50,
        use_temporal_ensemble=True,
    ) == 1.0
