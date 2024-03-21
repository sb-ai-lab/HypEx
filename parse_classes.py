import ast
from pathlib import Path


def get_class_attributes(node):
    attributes = [
        attr.targets[0].id
        for attr in node.body
        if isinstance(attr, ast.Assign) and not attr.targets[0].id.startswith("_")
    ]
    print(attributes)
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
            if isinstance(node, ast.ClassDef)
        }


def parse_directory(directory):
    classes = {}
    for file in Path(directory).rglob("*.py"):
        classes.update(parse_file(file))
    return classes


classes = parse_directory("hypex")
print(classes)
