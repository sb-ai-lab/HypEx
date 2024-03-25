import ast
import json
from pathlib import Path

unnecessary_dirs = ("dataset", "factory", "errors", "experiment", "hypotheses", "utils")
unnecessary_fields = ("loc", "iloc")


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
            for call in func.body:
                if isinstance(call, ast.Assign):
                    for target in call.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and target.value.id == "self"
                            and not (
                                target.attr.startswith("_")
                                or target.attr.startswith("default")
                            )
                        ):
                            required.append(target.attr)
                            objects[target.attr] = {
                                "type": "",
                                "default": {},
                                "title": f"The {target.attr} Schema",
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
        if len(file.parent.parts) > 1 and file.parent.parts[1] not in unnecessary_dirs:
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
