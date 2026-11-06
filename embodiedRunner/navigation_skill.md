# Navigation Skill

Version: navigation-seed/1. This is an initial authored strategy, not a learned
or experimentally validated Skill.

Use the current first-person image to identify the target described by the
instruction. Resolve its color, appearance, or commonsense description before
committing to a direction. A missing detection or an occluded view does not
establish that the object is absent.

When the target is visible, choose short movements consistent with its location
in the current image and the visible free space. Use at most two actions before
observing again. Do not infer metric travel distance from apparent image size.
The available sideways moves preserve camera heading, while rotations change it.

When the target is outside the current view, inspect other directions using the
available rotations or camera tilts. Retain an account of views and failed
actions to avoid blindly repeating a blocked move or cycling through the same
view without progress.

After a collision or another invalid action, discard the remaining plan and use
the new image and feedback to choose a different action. Object positions and
relative directions may change after every movement. Plan from the latest
observation rather than reusing stale coordinates.

Select only the published action IDs, copying their names exactly. Let the
environment determine task completion; do not create a stop action, assert
success, or use inaccessible target coordinates.
