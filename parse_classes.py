import ast
import json
from pathlib import Path

unnecessary_classes = ("ExperimentData", "")
unnecessary_fields = ("loc", "iloc")


def get_class_attributes(node):
    attributes = [
        attr.targets[0].id
        for attr in node.body
        if isinstance(attr, ast.Assign) and not attr.targets[0].id.startswith("_")
    ]
    for func in node.body:
        if isinstance(func, ast.FunctionDef) and func.name == "__init__":
            for call in func.body:
                if isinstance(call, ast.Assign):
                    attributes.extend(
                        [
                            target.attr
                            for target in call.targets
                            if isinstance(target, ast.Attribute)
                            and target.value.id == "self"
                            and not target.attr.startswith("_")
                        ]
                    )
    return attributes


def parse_file(file_path):
    with open(file_path, "r") as source_code:
        tree = ast.parse(source_code.read())
        return {
            node.name: get_class_attributes(node)
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name not in unnecessary_classes
        }


def parse_directory(directory):
    classes = {}
    for file in Path(directory).rglob("*.py"):
        if file.name != "roles.py" and file.name != "pandas_backend.py":
            classes.update(parse_file(file))
    return classes


def make_json():
    classes = parse_directory("hypex")
    with open("hypex/hypotheses/experiment_scheme.json", "w") as file:
        json.dump(classes, file, indent=4)


make_json()
