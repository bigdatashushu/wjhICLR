# BLINK multi-view reasoning with an explicit evidence round

Use the supplied original images, image order, question and option labels. Anchor every entity, landmark and pixel to its view_index. Image left/right, an observer's left/right and clockwise/counter-clockwise describe different frames; establish the question's frame before selecting an option.

## First solve round: inspect the available evidence

On the FIRST solve request, call ctx.tools.view_depth_summary() and inspect its standard response. If response['status'] == 'ok', retain response['result_id']. YieldObservations with the successful result IDs so the next reasoning request receives the actual data. Do not end this first round with a preselected ReturnAnswer. This inventory establishes which original views have usable predicted depth; it is not a measured camera-motion result.

Use this complete first-round program when no additional grounded query is needed:

```python
def solve(ctx):
    response = ctx.tools.view_depth_summary()
    if response['status'] == 'ok':
        YieldObservations([response['result_id']])
```

The first-round rule applies when no valid observations have yet been supplied. Once observations are present, use them and advance the reasoning instead of repeating the inventory. During host FINALIZATION, follow the requested JSON answer format and make no tool calls.

## Choose a measurement that answers a specific uncertainty

After the inventory, identify the visual landmarks and the competing interpretations. Decide which same-view measurement, if any, can distinguish them.

- Use view_candidates(view_index=..., category=...) when a named category would help locate a visible object. An empty list means no detected candidate, not no object. Category confirmation does not establish instance identity, mask purity or cross-view correspondence. Match landmarks by appearance, visible faces and neighboring context.
- Use point_depth(view_index, x, y) for a visible landmark whose near/far ordering matters to occlusion, scene layout or a parallax interpretation. Ground x,y in the original image dimensions. Sample visible surfaces, away from boundaries, background and occluders; use several justified points when one surface sample is ambiguous. Compare supported depths only within that view.
- Use pair_point_distance(view_index, a_xy, b_xy) when two competing interpretations differ in a relevant spatial distance within the same image. Both points must be visibly grounded in that view. Image-plane spacing is not spatial distance.
- A depth inventory alone does not identify object depth or camera motion. When a geometric relation is relevant and supported pixels are available, obtain the targeted measurement before treating that relation as established. Do not invent pixel coordinates or perform an unrelated measurement merely to increase the tool count.

Batch the relevant calls into one solve program, collect the successful result IDs and YieldObservations. A tool call occurs during program execution; the model cannot see its output while drafting that program. Yield first when the interpretation depends on seeing the measurement. The following reasoning request or finalization will receive the returned observations.

Every response is a ModelToolResponse dictionary with outer status and nested data. Check response['status'] == 'ok'; read response['data']; yield response['result_id']. There is no outer boolean 'ok'. Failed calls provide no valid evidence. ReadObservation and candidate_depth_summary are unavailable in this pipeline. Use only the listed tools and the original image coordinates.

## Reason about camera motion through stable landmarks

For a static scene, match at least two distinct landmarks when visible. Describe which physical face or side is exposed in each image and how neighboring landmarks or occlusions change. Keep landmark correspondence explicit instead of treating a shared category as the same instance.

Distinguish camera rotation, camera translation and movement around an object. A horizontal image displacement can arise from several motions and does not determine orbital direction. Near/far ordering from point_depth may inform a same-view parallax interpretation, but it does not prove a camera trajectory.

Compare the two candidate directions in the question's viewing convention. Check which direction explains the observed change in visible faces and at least one independent landmark or occlusion relation. Avoid guessing an object's semantic front when its appearance does not establish it. Separate intrinsic front/back from image left/right.

Before selecting the option, check that reversing the image order would reverse the inferred movement under the same viewing convention. Recheck the stated option mapping after reasoning; do not assume that a direction always corresponds to the same answer letter.

## Use the evidence in its actual coordinate system

Depths and point distances are relative predictions within independent camera views. They are not meters or a shared world frame. Never subtract depths between images, compare distances across images, or derive a camera transform from independent point maps. A same-view depth ordering may constrain a local interpretation; cross-view identity and orientation must still come from visibly matched landmarks and the stated axes.

When the question depends only on cross-view orientation and the available same-view measurements cannot distinguish the options, complete the evidence inventory, explain the remaining inference through visible landmarks, and return a visual estimate. Do not invent a geometric conclusion to make a tool appear useful.

On a later solve round, end with ReturnAnswer(AnswerPayload(value=selected_option, unit='option', basis='visual_estimate')) when the choice rests on visual reasoning, or basis='mixed' when an actual targeted measurement contributes to the choice. A depth-availability check alone is not a tool-derived solution. ReturnAnswer and YieldObservations terminate the cell immediately; place all intended work before the selected control.

Keep programs short and limited to the chosen queries and controls. Use existing observations rather than restarting a completed reasoning step. The host bounds the solve rounds and finalization; obey the current phase and return one supplied option.
