"""Compact task-card protocol and deterministic, verified public example excerpts."""
from functools import lru_cache
import json

from protocol import canonical, sha256

GUIDANCE = """Return JSON with subgoal={id,goal,object,destination,complete_when},
transition={kind,source,observation_revision,detail},progress={completed,remaining,next},
and ONE of program_source,perception_request,catalog_request. Assess progress in THIS SAME call.
progress strings<=240; model beliefs, not official facts. goal<=160, other subgoal strings<=80;
object/destination may be empty. complete_when: holding,released,navigation_ack,action_ack,visual.
kind: keep,advance,revise; source: none,official_feedback,visual; detail<=160.
transition.observation_revision MUST be an INTEGER copied from identity.observation_revision.
Never write the string "current", a quoted number, or a stale revision.
Initialize keep/none. Keep id/object/destination/complete_when stable; revise changes targets.
Advance evaluates PREVIOUS goal. destination=current waypoint; preserve ultimate task.
Matching Find ACK proves location, not holding; motion ACK does not prove arrival. Obey holding facts.
Prefer direct action. perception_request={targets:[{role:"object",category:"chair"}],depth:false}:
<=1/decision,0..4 distinct lowercase English noun phrases<=64; role object/destination/support.
depth is a JSON boolean. category must be lowercase ("bread", not "Bread"); targets or depth needed.
Empty detections do not prove absence. catalog_request={offset:0}; full catalog already supplied.
Each completion costs a call. Obey remaining limits and loop.switch_strategy/cycle_next_action.
program_source defines solve(ctx); host calls it. Globals injected; no imports/IO/reset/step.
ctx.action_table is a tuple of original NAME STRINGS; zero-based indices are action IDs.
Read ctx.instruction,feedback,memory,capabilities,task_state directly; never mutate. Subgoal is synchronized.
def solve(ctx):
    SubmitActions(ActionPayload(ctx.decision_id,ctx.observation_revision,ctx.snapshot_sha256,ctx.action_table_sha256,[0],[]))
Choose actual ID; only first action executes. Empty actions stops; use only when done. No automatic retry.
Tools return {result_id,data}; view_candidates(view_index=-1,category=''): data.candidates=list,count,coverage.
view_depth_summary(),point_depth(view_index,x,y),pair_point_distance(view_index,a_xy,b_xy): SAME-view relative.
candidate_depth_summary requires validated mask. ReadObservation(result_id,fields=None,offset=0,limit=8)
reads current results; YieldObservations(result_ids,reason='') ends segment. Reference exposed result_ids.
Results expire after mutation; fresh namespace each segment. Current RGB and complete Skill follow.
Complete output example for revision 0 (replace 0 with this request's integer revision and actual action):
{"subgoal":{"id":"locate","goal":"Locate the target","object":"","destination":"","complete_when":"visual"},
"transition":{"kind":"keep","source":"none","observation_revision":0,"detail":""},
"progress":{"completed":"","remaining":"Reach target","next":"Move toward target"},
"program_source":"def solve(ctx):\\n    SubmitActions(ActionPayload(ctx.decision_id,ctx.observation_revision,ctx.snapshot_sha256,ctx.action_table_sha256,[0],[]))"}
"""


@lru_cache(maxsize=16)
def examples(environment, subset):
    """Select whole source lines, retaining original action names and feedback."""
    if environment == "eb_navigation":
        from official_bare import literal
        rows, digest = literal("embodiedbench/evaluator/config/eb_navigation_example.py", "examples")
    elif environment in {"eb_alfred", "eb_habitat"}:
        from household_protocol import source
        name = ("alfred_long_horizon_examples.json" if subset == "long_horizon"
                else "alfred_examples.json") if environment == "eb_alfred" else "habitat_examples.json"
        data = source("embodiedbench/evaluator/config/" + name)
        rows, digest = json.loads(data), sha256(data)
    else:
        return ()
    selected = []
    for index, original in enumerate(rows):
        lines = []
        for line in original.splitlines():
            clean = " ".join(line.split())
            if clean.startswith(("Human instruction:", "Human Instruction", "Executable plans:",
                                 "Environment Feedback:")):
                lines.append(clean)
            elif '"action_id"' in clean:
                lines.append(clean)
        excerpt = "\n".join(lines)
        # Never cut a long source example. Select a smaller complete excerpt.
        if excerpt and len(excerpt) <= 1100:
            selected.append({"source_sha256": digest, "source_index": index,
                             "excerpt": excerpt})
        if len(selected) == 2:
            break
    # Navigation tutorials contain many stages. Retain one complete public
    # action/feedback cycle if no entire line-only excerpt fits.
    if not selected and rows:
        lines = [" ".join(x.split()) for x in rows[0].splitlines() if x.strip()]
        start = next(i for i, x in enumerate(lines) if x.startswith("Executable plans:"))
        end = next(i for i in range(start + 1, len(lines))
                   if lines[i].startswith("Environment Feedback:"))
        selected.append({"source_sha256": digest, "source_index": 0,
                         "excerpt": "\n".join([lines[0], *lines[start:end + 1]])})
    # 1500 is an optional-section target; reduce by whole examples.
    while selected and len(canonical(selected).decode()) > 1500:
        selected.pop()
    return tuple(selected)


def render(environment, contract, revision=None):
    text = GUIDANCE
    if environment == "eb_manipulation":
        text = text.replace(
            "ctx.action_table is a tuple of original NAME STRINGS; zero-based indices are action IDs.",
            "ctx.action_table is empty. Also available: ctx.action_spec,ctx.official_objects,ctx.task_variation. "
            "Actions are [x,y,z,roll,pitch,yaw,gripper]: position integers 0..100, "
            "rotation integers 0..120 (3 degrees), gripper 0/1. Official objects use world voxels; "
            "predicted relative depth cannot convert to world voxels.")
        text = text.replace("[0],[]", "[[50,50,30,120,60,90,1]],[]")
    if revision is not None:
        text += f"\nFor THIS request, transition.observation_revision MUST be the integer {revision}.\n"
    return "Execution contract " + contract.contract_version + "; sha256=" + contract.sha256 + "\n" + text
