"""Audit every applicable exported pin, including alternative extra stacks."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def audit_batches(lines):
    """Partition pins by package name without dropping alternative versions."""
    batches = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        specifiers = list(requirement.specifier)
        if (
            requirement.url
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(f"Expected an exact version pin: {line}")
        name = canonicalize_name(requirement.name)
        pin = f"{name}=={specifiers[0].version}"
        if pin in seen:
            continue
        seen.add(pin)
        for batch in batches:
            if name not in batch:
                break
        else:
            batch = {}
            batches.append(batch)
        batch[name] = pin
    return [list(batch.values()) for batch in batches]


def main():
    batches = audit_batches(Path(sys.argv[1]).read_text().splitlines())
    failed = False
    with TemporaryDirectory(prefix="hypex-audit-") as directory:
        for index, batch in enumerate(batches, start=1):
            requirements = Path(directory) / f"requirements-{index}.txt"
            requirements.write_text("\n".join(batch) + "\n")
            print(
                f"Auditing batch {index}/{len(batches)} ({len(batch)} pins)", flush=True
            )
            # Poetry already supplies transitive pins. Alternative extra stacks
            # cannot be installed together, so bypass pip's installation step.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--no-deps",
                    "--disable-pip",
                    "--requirement",
                    str(requirements),
                    *sys.argv[2:],
                ],
                check=False,
            )
            failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
