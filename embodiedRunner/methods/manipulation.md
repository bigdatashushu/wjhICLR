# EB-Manipulation: current-view gripper control

Use the current instruction, annotated front image, official anonymous object coordinates and action feedback to choose the next gripper motion. Objects are numbered by ascending world Y. Interpret appearance in the image before choosing the relevant object; its number does not identify a target.

The runtime exposes `ctx.official_objects`, `ctx.action_spec` and `ctx.task_variation`. Each action is `[x, y, z, roll, pitch, yaw, gripper]`: position integer indices 0–100, Euler integer indices 0–120 at 3 degrees per index, and gripper 0 for closed or 1 for open. There are no action IDs. Submit an action list through the injected `ActionPayload` and `SubmitActions`; the host executes its first action and gives a new observation.

For picking, approach above the selected object with an open gripper, descend to a reachable grasp, close, then lift. For stacking or placing, retain the grasp during transport and open at the intended placement. For wiping, identify the instructed region and move the grasped sponge along its main direction. Use current feedback to adjust approach height or orientation when a motion fails. Avoid repeating an unchanged rejected motion.

Maintain a concise subgoal and sourced history of what was attempted. A new step invalidates all previous current evidence, including when the camera image stays the same. A previous intended grasp is not evidence that the gripper holds the object; inspect feedback and current appearance.

Official anonymous coordinates are world voxel inputs produced by the benchmark's depth, instance masks and camera parameters. They are separate from the prediction tools. `point_depth` and `pair_point_distance` read original current image predictions in relative units; they cannot convert image pixels into world voxel gripper coordinates. Candidate category confirmation does not validate a mask surface. Use tool results only within their stated scope.

Submit an empty list only when choosing to stop. Task success and termination are determined by the official environment.
