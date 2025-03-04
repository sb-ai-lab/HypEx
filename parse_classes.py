import ast
import json
from pathlib import Path

unnecessary_dirs = (
    "dataset",
    "factory",
    "errors",
    "experiment",
    "hypotheses",
    "utils",
    "executor",
)


def get_class_attributes(node):
    required = []
    objects = {}
    for func in node.body:
        if isinstance(func, ast.Assign) and not (
            func.targets[0].id.startswith("_")
            or func.targets[0].id.startswith("default")
        ):
            required.append(func.targets[0].id)
            objects[func.targets[0].id] = {
                "type": "",
                "default": {},
                "title": f"The {func.targets[0].id} Schema",
            }
        if isinstance(func, ast.FunctionDef) and func.name == "__init__":
            for arg in func.args.args:

                if isinstance(func.args.defaults[0], ast.AST) and arg.arg not in (
                    "self",
                    "key",
                    "full_name",
                    "transformer",
                ):
                    required.append(arg.arg)
                    objects[arg.arg] = {
                        "type": "",
                        "default": {},
                        "title": f"The {arg.arg} Schema",
                    }
    return {"required": required, "properties": objects}


def parse_file(file_path):
    with open(file_path, "r") as source_code:
        tree = ast.parse(source_code.read())
        return {
            node.name: get_class_attributes(node)
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }


def parse_directory(directory):
    classes = {}
    for file in Path(directory).rglob("*.py"):
        if len(file.parent.parts) > 1 and (
            (
                file.parent.parts[1] not in unnecessary_dirs
                and file.name != "abstract.py"
            )
            or file.parent.parts[1] == "experiments"
            and file.name == "base.py"
        ):
            classes.update(parse_file(file))
    return classes


def make_json():
    classes = dict(sorted(parse_directory("hypex").items()))
    template = {
        "type": "object",
        "default": {},
        "title": "Root Schema",
        "required": ["experiment"],
        "properties": {
            "experiment": {
                "type": "object",
                "default": {},
                "title": "The experiment Schema",
                "properties": classes,
            }
        },
    }
    with open("hypex/hypotheses/experiment_scheme.json", "w") as file:
        json.dump(template, file, indent=4)


make_json()
