from pathlib import Path
import re
from typing import List

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent.resolve()


def read_version() -> str:
    init_file = ROOT / "jea" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_file.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError("Could not find __version__ in jea/__init__.py")
    return match.group(1)


def read_requirements() -> List[str]:
    requirements = []
    for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="jea",
    version=read_version(),
    description="The Locality Trade-off in Masked Patch Self-Distillation",
    packages=find_namespace_packages(include=("jea", "jea.*"), exclude=("*.__pycache__",)),
    include_package_data=True,
    package_data={"jea": ["configs/*.yaml", "configs/train/*/*.yaml"]},
    install_requires=read_requirements(),
    python_requires=">=3.10",
)
