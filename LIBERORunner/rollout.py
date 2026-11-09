"""Sequential equivalent of original evaluate_one_task_success's control loop."""
from __future__ import annotations

from .policy import InvalidAction


def rollout(env, policy, initial_state, language: str, max_steps: int,
            settling_steps: int, journal) -> dict:
    env.reset()
    obs = env.set_init_state(initial_state)
    journal({"event": "initial_state_set"})
    # Original LIBERO metric uses all-zero actions, including gripper=0.
    for i in range(settling_steps):
        journal({"event": "settling_action_started", "step": i, "action": [0.0] * 7})
        obs, _, _, _ = env.step([0.0] * 7)
        journal({"event": "settling_action_ack", "step": i})
    ack_count = 0
    for step in range(max_steps):
        try:
            action = policy.act(obs, language, step)
        except InvalidAction:
            return {"status": "invalid_action", "success": False,
                    "control_steps": ack_count}
        journal({"event": "action_started", "step": step, "action": action})
        obs, reward, done, _ = env.step(action)
        success = bool(env.check_success())
        ack_count += 1
        journal({"event": "action_ack", "step": step, "action": action,
                 "environment_done": bool(done), "goal_satisfied": success,
                 "reward": float(reward)})
        # Original LIBERO replaces done with its BDDL success result.
        # Fail if a different wrapper or simulator starts terminating for time.
        if bool(done) != success:
            raise RuntimeError("Environment done and LIBERO goal result disagree.")
        if done:
            return {"status": "success", "success": True,
                    "control_steps": ack_count}
    return {"status": "task_failure", "success": False, "control_steps": ack_count}
