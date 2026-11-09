"""Read and verify frozen sources; never import upstream provider/simulator SDKs."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "9be4e980e9cd6bcb38373cd4aab7c32724bdd401"
TREES = {
    "navigation": ("upstream", "official_sources.json"),
    "alfred": ("upstream-household", "household_sources.json"),
    "habitat": ("upstream-household", "household_sources.json"),
    "manipulation": ("upstream-manipulation", "manipulation_sources.json"),
}
LIMITS = {"navigation": 20, "alfred": 30, "habitat": 30, "manipulation": 15}


class ContractError(ValueError):
    pass


class OperationalError(RuntimeError):
    def __init__(self, code, http_status=None):
        self.code, self.http_status = code, http_status
        super().__init__(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def implementation_hashes():
    return {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


class Sources:
    def __init__(self, root=ROOT, manipulation_catalog=None, manipulation_dataset=None):
        self.root = Path(root).resolve()
        self.manipulation_catalog = Path(manipulation_catalog or self.root / "manipulation_assets.json").resolve()
        self.manipulation_dataset = Path(manipulation_dataset or self.root / "manipulation-runtime/dataset").resolve()
        self.records, self.hashes = {}, {}
        for tree, manifest in set(TREES.values()):
            raw = (self.root / manifest).read_bytes()
            document = json.loads(raw)
            if document["revision"] != REVISION:
                raise ContractError("official_revision_mismatch")
            self.records[tree] = {row["path"]: row for row in document["files"]}
            self.hashes[manifest] = digest(raw)
        for manifest in ("action_alfred_data.json", "alfred_pilot_data.json"):
            self.hashes[manifest] = file_digest(self.root / manifest)
        self.hashes["manipulation_assets.json"] = file_digest(self.manipulation_catalog)

    def read(self, kind, path):
        tree = TREES[kind][0]
        row = self.records[tree].get(path)
        if row is None:
            raise ContractError("unlisted_official_source")
        location = (self.root / tree / path).resolve()
        if (self.root / tree).resolve() not in location.parents:
            raise ContractError("official_path_escape")
        data = location.read_bytes()
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ContractError("official_source_changed")
        self.hashes[tree + "/" + path] = row["sha256"]
        return data

    def literal(self, kind, path, variable):
        tree = ast.parse(self.read(kind, path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == variable for t in node.targets):
                return ast.literal_eval(node.value)
        raise ContractError("official_literal_missing")

    def verify_all(self):
        counts = {}
        for kind in ("navigation", "alfred", "manipulation"):
            tree = TREES[kind][0]
            for path in self.records[tree]:
                self.read(kind, path)
            counts[tree] = len(self.records[tree])
        return counts

    def planner_class(self, kind, namespace):
        path, name = {
            "navigation": ("nav_planner.py", "EBNavigationPlanner"),
            "alfred": ("vlm_planner.py", "VLMPlanner"),
            "habitat": ("vlm_planner.py", "VLMPlanner"),
            "manipulation": ("manip_planner.py", "ManipPlanner"),
        }[kind]
        tree = ast.parse(self.read(kind, "embodiedbench/planner/" + path))
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        selected = {"__init__", "process_prompt", "get_message", "reset", "update_info"}
        if kind != "manipulation":
            selected.update({"set_actions", "get_availabel_action_prompt"})
        if kind in ("alfred", "habitat"):
            selected.add("json_to_action")
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        if {n.name for n in methods} != selected:
            raise ContractError("official_prompt_methods_changed")
        constructor = next(n for n in methods if n.name == "__init__")
        providers = [n for n in constructor.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "model_type == 'custom'"]
        if len(providers) != 1:
            raise ContractError("official_provider_initialization_changed")
        # Remove only SDK construction. Prompt/feedback/parser methods stay verbatim.
        constructor.body.remove(providers[0])
        module = ast.fix_missing_locations(ast.Module(body=[
            ast.ClassDef(name=name, bases=[], keywords=[], body=methods, decorator_list=[])
        ], type_ignores=[]))
        exec(compile(module, path, "exec"), namespace)
        return namespace[name]
