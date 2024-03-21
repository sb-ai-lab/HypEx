import ast
import os


def parse_directory(directory):
    classes = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r") as source_code:
                    tree = ast.parse(source_code.read())
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            attributes = [
                                attr.targets[0].id
                                for attr in node.body
                                if isinstance(attr, ast.Assign)
                            ]

                            for func in node.body:
                                if (
                                    isinstance(func, ast.FunctionDef)
                                    and func.name == "__init__"
                                ):
                                    for call in func.body:
                                        if isinstance(call, ast.Assign):
                                            attributes.extend(
                                                [
                                                    target.attr
                                                    for target in call.targets
                                                    if isinstance(target, ast.Attribute)
                                                    and target.value.id == "self" and target.value.
                                                ]
                                            )

                            classes[node.name] = attributes

    return classes


classes = parse_directory("hypex")
print(classes)
