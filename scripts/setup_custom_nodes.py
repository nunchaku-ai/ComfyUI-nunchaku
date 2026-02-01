#!/usr/bin/env python3
"""
Script to clone custom nodes from test_data/custom_nodes.yaml,
checkout to specified commits, and install dependencies.
"""

import subprocess
import sys
from pathlib import Path

import yaml


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run（cmd， cwd=cwd， capture_output=True， text=True， check=check， encoding=“utf-8”， errors=“replace”）
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def clone_and_setup_node(node_info, custom_nodes_dir, dependencies_dir):
    """Clone a node repository, checkout to specified commit, and install dependencies."""
    name = node_info["name"]
    url = node_info["url"]
    branch = node_info.get("branch", "main")

    node_path = custom_nodes_dir / name

    print(f"\n{'='*60}")
    print(f"Setting up: {name}")
    print(f"{'='*60}")

    # Clone repository if it doesn't exist
    if not node_path.exists():
        print(f"Cloning {name} from {url}...")
        try:
            run_command(["git", "clone", url, str(node_path)])
        except subprocess.CalledProcessError as e:
            print(f"Error cloning {name}: {e}", file=sys.stderr)
            return False
    else:
        print(f"{name} already exists at {node_path}")

    # Checkout to specified commit/branch
    print(f"Checking out to {branch}...")
    try:
        run_command(["git", "fetch", "--all"], cwd=node_path)
        run_command(["git", "checkout", branch], cwd=node_path)
    except subprocess.CalledProcessError as e:
        print(f"Error checking out {branch} for {name}: {e}", file=sys.stderr)
        return False

    # Install dependencies
    print(f"Installing dependencies for {name}...")

    # Check if custom dependency file exists in test_data/dependencies
    custom_deps_file = dependencies_dir / f"{name}.txt"

    if custom_deps_file.exists():
        print(f"Using custom dependencies from {custom_deps_file}")
        try:
            run_command(["uv", "pip", "install", "-r", str(custom_deps_file)])
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies from {custom_deps_file}: {e}", file=sys.stderr)
            return False
    else:
        # Fall back to requirements.txt in the repository
        requirements_file = node_path / "requirements.txt"
        if requirements_file.exists():
            print(f"Using requirements.txt from repository: {requirements_file}")
            try:
                run_command(["uv", "pip", "install", "-r", str(requirements_file)])
            except subprocess.CalledProcessError as e:
                print(f"Error installing dependencies from {requirements_file}: {e}", file=sys.stderr)
                return False
        else:
            print(f"No requirements.txt found for {name}, skipping dependency installation")

    print(f"✓ Successfully set up {name}")
    return True


def main():
    # Get project root directory (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define paths
    custom_nodes_yaml = project_root / "test_data" / "custom_nodes.yaml"
    dependencies_dir = project_root / "test_data" / "dependencies"
    custom_nodes_dir = Path("custom_nodes")

    # Verify required files exist
    if not custom_nodes_yaml.exists():
        print(f"Error: {custom_nodes_yaml} not found!", file=sys.stderr)
        sys.exit(1)

    # Create custom_nodes directory if it doesn't exist
    custom_nodes_dir.mkdir(exist_ok=True)

    # Load custom nodes configuration
    print(f"Loading configuration from {custom_nodes_yaml}...")
    with open(custom_nodes_yaml, "r") as f:
        config = yaml.safe_load(f)

    nodes = config.get("nodes", [])
    if not nodes:
        print("No nodes found in configuration")
        sys.exit(0)

    print(f"Found {len(nodes)} node(s) to set up")

    # Process each node
    success_count = 0
    failed_nodes = []

    for node_info in nodes:
        success = clone_and_setup_node(node_info, custom_nodes_dir, dependencies_dir)
        if success:
            success_count += 1
        else:
            failed_nodes.append(node_info["name"])

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Successfully set up: {success_count}")
    print(f"Failed: {len(failed_nodes)}")

    if failed_nodes:
        print("\nFailed nodes:")
        for name in failed_nodes:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\n✓ All nodes set up successfully!")


if __name__ == "__main__":
    main()
