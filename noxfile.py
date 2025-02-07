"""Configuration for Nox."""

import argparse
import os
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["lint", "tests", "doctests"]
nox.options.default_venv_backend = "uv|virtualenv"

# ===================================================================
# Linting


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session
def pylint(session: nox.Session) -> None:
    """Run PyLint."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install(".", "pylint")
    session.run("pylint", "galax", *session.posargs)


# ===================================================================
# Testing


@nox.session
def tests_standard(session: nox.Session) -> None:
    """Run the regular tests: src, README, docs, tests/unit."""
    session.install("-e", ".[test]")
    os.environ["GALAX_ENABLE_RUNTIME_TYPECHECKS"] = "1"
    session.run("pytest", "src", "README", "docs", "tests/unit", *session.posargs)


@nox.session
def tests_all(session: nox.Session) -> None:
    """Run all the tests."""
    session.install("-e", ".[test-all]")
    os.environ["GALAX_ENABLE_RUNTIME_TYPECHECKS"] = "1"
    session.run("pytest", *session.posargs)


@nox.session
def doctests(session: nox.Session) -> None:
    """Run the doctests: README, docs, src -- including mpl tests."""
    session.install(".[test,test-mpl]")
    os.environ["GALAX_ENABLE_RUNTIME_TYPECHECKS"] = "1"
    session.run(
        "pytest",
        *("README", "docs", "src/galax"),
        "--mpl",
        *session.posargs,
    )


@nox.session
def generate_mpl_tests(session: nox.Session) -> None:
    """Generate the mpl tests."""
    session.install(".[test,test-mpl]")
    os.environ["GALAX_ENABLE_RUNTIME_TYPECHECKS"] = "1"
    session.run(
        "pytest",
        "tests",
        "-m mpl_image_compare",  # only run the mpl tests
        "--mpl-generate-hash-library=tests/mpl_figure/hashes.json",
        "--mpl-generate-path=tests/mpl_figure/baseline",
        "--mpl-generate-summary=html,json,basic-json",
        *session.posargs,
    )


@nox.session
def test_mpl(session: nox.Session) -> None:
    """Test the figures."""
    session.install(".[test,test-mpl]")
    os.environ["GALAX_ENABLE_RUNTIME_TYPECHECKS"] = "1"
    session.run(
        "pytest",
        "tests",
        "--mpl",
        "-m mpl_image_compare",  # only run the mpl tests
        "--mpl-generate-summary=basic-html,json",
        *session.posargs,
    )


# ===================================================================


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.install("-e.[docs]", *extra_installs)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session) -> None:
    """Build (regenerate) API docs."""
    session.install("sphinx")
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/galax",
    )


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
