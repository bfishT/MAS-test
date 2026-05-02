#!/usr/bin/env python3
"""
SWE-bench + LangGraph Agent Pipeline (Docker-based)

Pipeline: Load SWE-bench instances → Pull pre-built Docker image →
          Generate DAG plan via LLM → Build LangGraph with tool-calling
          agent nodes → Execute inside container → Extract patch →
          Output SWE-bench prediction format.

Usage:
    python run_with_langgraph.py \
        --dataset_name_or_path princeton-nlp/SWE-bench_oracle \
        --instance_ids django__django-16379 \
        --model glm-5.1 \
        --api-key YOUR_KEY \
        --base-url https://opencode.ai/zen/go/v1 \
        --output_dir ./outputs

    # Then evaluate with SWE-bench harness
    python -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_oracle \
        --predictions_path outputs/glm-5.1__SWE-bench_oracle__test.jsonl \
        --max_workers 4 \
        --run_id glm5_dag_test
"""

import json
import os
import re
import time
import argparse
import traceback
from collections import defaultdict
import io
import tarfile
from pathlib import Path
from typing import TypedDict, Annotated

import docker
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from datasets import load_dataset, load_from_disk

# ─── Prompts ───────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are a software engineering planning assistant. Given a bug report or feature request (an "issue"), your job is to produce a step-by-step plan to resolve it, represented as a Directed Acyclic Graph (DAG).

Output format rules — follow them EXACTLY:
0. The Plan should as Efficient as Possible, as Parallelly as Possible
1. First section: node definitions, one per line, in the format:
   LETTER: description
   where LETTER is a single uppercase letter (A, B, C, …). Use consecutive letters starting from A.
2. Second section: a blank line to separate the two sections.
3. Third section: edge definitions, one per line, in the format:
   LETTER -> LETTER
   representing that the left node must be completed before the right node can start.
4. The graph MUST be a valid DAG: no cycles, every node (except roots) must have at least one incoming edge, and every node (except leaves) must have at least one outgoing edge.
5. The plan should be concrete and actionable: each node describes a specific engineering step (e.g. "Identify root cause in module X", "Write test case reproducing the bug", "Implement fix in module Y", "Run test suite to verify").
6. IMPORTANT: Different nodes MUST NOT modify the same file. If two steps need to edit the same file, they must be sequential (one depends on the other).
7. Output NOTHING else besides the DAG — no markdown, no explanation, no preamble.
"""

AGENT_SYSTEM_PROMPT = """\
You are an autonomous software engineering agent working inside a Docker container with a full development environment. The repository is already cloned and installed at /testbed. All dependencies are ready.

You have access to tools for searching, reading, editing, and executing code.

IMPORTANT RULES:
- The codebase is at /testbed. Use RELATIVE paths from /testbed.
- The environment is fully set up — do NOT install packages or clone repos.
- Use the `run_test` tool to run tests. Do NOT use pip install.
- After making changes, use the `run_test` tool to verify your work before finishing.
- Make minimal, targeted changes — prefer `edit_file` over `write_file` to avoid breaking other code.
- When you are done, output a final summary of what you changed and why.
- Do NOT repeat investigations that were already done in previous steps — read the "Previously Completed Steps" section carefully.
- Focus on YOUR specific task only. Do NOT waste rounds on unrelated exploration.
"""

# ─── Tool Definitions ─────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a regex pattern in the codebase. Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (relative to /testbed, default '.')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to /testbed)",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based, optional)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Line number to end reading at (inclusive, optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing a specific text snippet. This is safer than write_file as it only changes the targeted portion. The old_text must match exactly (including whitespace and indentation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to /testbed)",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to be replaced (must match exactly)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write full content to a file (overwrites existing). Use edit_file instead when possible to avoid unintended changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to /testbed)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory. Returns files and subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative to /testbed, default '.')",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "Run specific test cases to verify the fix. Use this after making changes to check if the bug is resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_command": {
                        "type": "string",
                        "description": "Test command to run (e.g. 'pytest tests/test_foo.py::test_bar')",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["test_command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the workspace. Use for running tests, git, python scripts, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

# ─── Docker-based Tool Execution ──────────────────────────────────────

DOCKER_WORKDIR = "/testbed"
DOCKER_USER = "root"
UTF8 = "utf-8"

HEREDOC_DELIMITER = "EOF_5f8a3b2c1d"


def _write_to_container(container, content: str, container_path: str):
    tar_stream = io.BytesIO()
    data = content.encode("utf-8")
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tarinfo = tarfile.TarInfo(name=os.path.basename(container_path))
        tarinfo.size = len(data)
        tar.addfile(tarinfo, io.BytesIO(data))
    tar_stream.seek(0)
    container.put_archive(os.path.dirname(container_path), tar_stream)


def docker_exec(
    container,
    cmd: str,
    workdir: str = DOCKER_WORKDIR,
    user: str = DOCKER_USER,
    timeout: int = 120,
) -> str:
    activated_cmd = f"source activate testbed 2>/dev/null; cd {workdir} && {cmd}"
    try:
        exit_code, output = container.exec_run(
            ["bash", "-c", activated_cmd],
            workdir=workdir,
            user=user,
        )
        text = output.decode(UTF8, errors="replace") if output else ""
        if exit_code != 0 and text:
            text += f"\n[exit code: {exit_code}]"
        return text
    except Exception as e:
        return f"Error: {e}"


def docker_exec_with_timeout(container, cmd: str, timeout: int = 120) -> str:
    activated_cmd = f"source activate testbed 2>/dev/null; cd {DOCKER_WORKDIR} && {cmd}"
    try:
        exec_id = container.client.api.exec_create(
            container.id,
            ["bash", "-c", activated_cmd],
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )
        output_chunks = []
        start = time.time()
        for chunk in container.client.api.exec_start(exec_id["Id"], stream=True):
            output_chunks.append(chunk.decode(UTF8, errors="replace"))
            if time.time() - start > timeout:
                return "".join(output_chunks)[:8000] + f"\n[TIMEOUT after {timeout}s]"
        result = "".join(output_chunks)
        return result[:8000] if result else "(no output)"
    except Exception as e:
        return f"Error: {e}"


def execute_edit_file(container, fpath: str, old_text: str, new_text: str) -> str:
    python_script = f"""
import sys
path = "/testbed/{fpath}"
with open(path, "r") as f:
    content = f.read()
if {repr(old_text)} not in content:
    print("ERROR: old_text not found in file. Make sure the text matches exactly.")
    sys.exit(1)
count = content.count({repr(old_text)})
if count > 1:
    print(f"WARNING: old_text found {{count}} times. Replacing first occurrence only.")
new_content = content.replace({repr(old_text)}, {repr(new_text)}, 1)
with open(path, "w") as f:
    f.write(new_content)
print(f"Successfully edited {{path}}")
"""
    escaped = python_script.replace("'", "'\\''")
    cmd = f"python3 -c '{escaped}'"
    return docker_exec(container, cmd)


def execute_tool(name: str, arguments: dict, container) -> str:
    if name == "search_code":
        pattern = arguments["pattern"]
        search_path = arguments.get("path", ".")
        cmd = f'grep -rn -E "{pattern}" {search_path} --include="*.py" --include="*.js" --include="*.ts" --include="*.java" --include="*.c" --include="*.h"'
        return docker_exec(container, cmd)

    elif name == "read_file":
        fpath = arguments["path"]
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line", "")
        if end_line:
            cmd = f"sed -n '{start_line},{end_line}p' {fpath} | cat -n"
        else:
            cmd = f"cat -n {fpath}"
        return docker_exec(container, cmd)

    elif name == "edit_file":
        fpath = arguments["path"]
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        return execute_edit_file(container, fpath, old_text, new_text)

    elif name == "write_file":
        fpath = arguments["path"]
        content = arguments["content"]
        try:
            _write_to_container(container, content, f"/testbed/{fpath}")
            return f"Successfully wrote {len(content)} chars to {fpath}"
        except Exception as e:
            return f"Error writing file: {e}"

    elif name == "list_directory":
        dir_path = arguments.get("path", ".")
        return docker_exec(container, f"ls -F {dir_path}")

    elif name == "run_test":
        test_cmd = arguments.get("test_command") or arguments.get("command", "")
        timeout = arguments.get("timeout", 120)
        return docker_exec_with_timeout(container, test_cmd, timeout=timeout)

    elif name == "run_command":
        cmd = arguments["command"]
        timeout = arguments.get("timeout", 120)
        return docker_exec_with_timeout(container, cmd, timeout=timeout)

    return f"Unknown tool: {name}"


# ─── DAG Plan Generation & Parsing ────────────────────────────────────


def generate_dag(client: OpenAI, model: str, question: str) -> str | None:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the issue:\n\n{question}"},
        ],
        temperature=0.3,
        max_tokens=10240,
    )
    content = response.choices[0].message.content
    if content is None:
        print(
            f"    [Warning] LLM returned None content. Finish reason: {response.choices[0].finish_reason}"
        )
        return None
    if not content.strip():
        print(
            f"    [Warning] LLM returned empty content. Finish reason: {response.choices[0].finish_reason}"
        )
        return None
    return content.strip()


def parse_dag(dag_text: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    node_section = True

    cleaned = re.sub(r"```[\w]*\n?", "", dag_text)

    for line in cleaned.strip().split("\n"):
        line = line.strip()
        if not line:
            if node_section and nodes:
                node_section = False
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        if node_section:
            if ":" not in line:
                continue
            letter, _, desc = line.partition(":")
            letter, desc = letter.strip().lstrip("-").strip(), desc.strip()
            if len(letter) == 1 and letter.isalpha():
                nodes[letter] = desc
        else:
            if "->" not in line and "→" not in line:
                continue
            if "→" in line:
                parts = [p.strip() for p in line.split("→")]
            else:
                parts = [p.strip() for p in line.split("->")]
            if len(parts) == 2 and all(len(p) == 1 and p.isalpha() for p in parts):
                edges.append((parts[0], parts[1]))

    return nodes, edges


def validate_dag(nodes: dict[str, str], edges: list[tuple[str, str]]) -> list[str]:
    issues: list[str] = []
    if not nodes:
        issues.append("No nodes defined")
        return issues
    for src, dst in edges:
        if src not in nodes:
            issues.append(f"Edge references undefined node: {src}")
        if dst not in nodes:
            issues.append(f"Edge references undefined node: {dst}")
    if not edges and len(nodes) > 1:
        issues.append("Multiple nodes but no edges defined")

    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        if src in adj and dst in adj:
            adj[src].append(dst)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def _has_cycle(node: str) -> bool:
        color[node] = GRAY
        for nb in adj.get(node, []):
            if color[nb] == GRAY:
                return True
            if color[nb] == WHITE and _has_cycle(nb):
                return True
        color[node] = BLACK
        return False

    for n in nodes:
        if color[n] == WHITE and _has_cycle(n):
            issues.append("Graph contains a cycle")
            break

    return issues


def fallback_single_node_dag(
    question: str,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    return {"A": "Investigate and fix the issue described in the bug report"}, []


# ─── SWE-bench Dataset Loading ────────────────────────────────────────


def load_swebench_dataset(
    path: str, split: str = "test", instance_ids: list[str] | None = None
) -> list[dict]:
    if Path(path).exists():
        if path.endswith(".jsonl") or path.endswith(".json"):
            instances = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        instances.append(json.loads(line))
        else:
            dataset = load_from_disk(path)
            if split in dataset:
                dataset = dataset[split]
            instances = [dict(row) for row in dataset]
    else:
        dataset = load_dataset(path)
        if split not in dataset:
            raise ValueError(f"Invalid split {split} for dataset {path}")
        dataset = dataset[split]
        instances = [dict(row) for row in dataset]

    if instance_ids:
        id_set = set(instance_ids)
        instances = [inst for inst in instances if inst["instance_id"] in id_set]

    print(f"Loaded {len(instances)} instances from {path}")
    return instances


# ─── Container Management ──────────────────────────────────────────────


def get_docker_image(instance_id: str) -> str:
    image_tag = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{image_tag}:latest"


def start_container(instance: dict):
    instance_id = instance["instance_id"]
    image = get_docker_image(instance_id)
    base_commit = instance.get("base_commit", "")

    docker_client = docker.from_env()
    print(f"  Pulling image {image}...")
    try:
        docker_client.images.get(image)
        print(f"  Image {image} already exists locally")
    except docker.errors.ImageNotFound:
        docker_client.images.pull(image)
        print(f"  Pulled {image}")

    print(f"  Starting container for {instance_id}...")
    container = docker_client.containers.run(
        image=image,
        command="sleep infinity",
        detach=True,
        working_dir="/testbed",
        mem_limit="8g",
        stdin_open=True,
        tty=True,
    )

    if base_commit:
        exit_code, output = container.exec_run(
            ["bash", "-c", f"cd /testbed && git checkout {base_commit}"],
            user="root",
        )
        if exit_code != 0:
            print(
                f"  [Warning] git checkout failed: {output.decode('utf-8', errors='replace')[:200]}"
            )
        exit_code, output = container.exec_run(
            ["bash", "-c", "cd /testbed && git clean -fd"],
            user="root",
        )

    python_info = container.exec_run(
        "python -c 'import sys; print(sys.version)'", workdir="/testbed"
    )
    print(f"  Container started: {container.name}")
    if python_info[0] == 0:
        print(
            f"  Container Python: {python_info[1].decode('utf-8', errors='replace').strip()[:60]}"
        )

    return container, docker_client


def connect_container(container_name: str):
    docker_client = docker.from_env()
    container = docker_client.containers.get(container_name)

    print(f"  Connected to container: {container.name}")
    result = docker_exec(container, "python -c 'import sys; print(sys.version)'")
    print(f"  Container Python: {result.strip()[:60]}")

    return container, docker_client


def extract_patch_from_container(container, base_commit: str = "") -> str:
    if base_commit:
        docker_exec(container, "git add -A")
        result = docker_exec(
            container, f"git -c core.fileMode=false diff {base_commit}"
        )
        patch = result.strip()
        if patch:
            return patch + "\n" if not patch.endswith("\n") else patch
    docker_exec(container, "git add -A")
    result = docker_exec(container, "git -c core.fileMode=false diff --cached")
    patch = result.strip()
    if not patch:
        result = docker_exec(container, "git -c core.fileMode=false diff")
        patch = result.strip()
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch


# ─── LangGraph State & Agent ──────────────────────────────────────────


def merge_dicts(left: dict, right: dict) -> dict:
    merged = dict(left)
    for k, v in right.items():
        if k in merged and merged[k] != v:
            print(f"  [Warning] merge_dicts conflict on key '{k}': overwriting")
        merged[k] = v
    return merged


def keep_last(left: str, right: str) -> str:
    return right


class AgentState(TypedDict):
    question: str
    fail_to_pass: str
    node_results: Annotated[dict, merge_dicts]
    current_node: Annotated[str, keep_last]


MAX_TOOL_ROUNDS = 100


def run_agent_loop(
    llm_client: OpenAI,
    model: str,
    messages: list[dict],
    container,
) -> str:
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            return f"[API error: {e}]"

        msg = response.choices[0].message
        # Only keep essential fields to avoid API validation errors
        # (some providers reject extra fields like refusal, annotations, audio, etc.)
        msg_dict: dict = {"role": msg.role}
        if msg.content:
            msg_dict["content"] = msg.content
        # reasoning_content is required by some reasoning models (e.g. Kimi K2.6)
        if getattr(msg, "reasoning_content", None):
            msg_dict["reasoning_content"] = msg.reasoning_content
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            print(f"      [Tool] {fn_name}({json.dumps(fn_args, default=str)[:120]})")
            result = execute_tool(fn_name, fn_args, container)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "[Agent reached max tool rounds]"


def build_langgraph(
    llm_client: OpenAI,
    model: str,
    question: str,
    fail_to_pass: str,
    nodes: dict[str, str],
    edges: list[tuple[str, str]],
    container,
) -> StateGraph:
    def make_agent_node(node_id: str, node_desc: str):
        def node_fn(state: AgentState) -> dict:
            print(f"  [Agent] Node {node_id}: {node_desc}")

            prev_summary = ""
            for k, v in state.get("node_results", {}).items():
                prev_summary += f"Step {k}: {v[:300]}\n\n"

            fail_info = ""
            if state.get("fail_to_pass"):
                fail_info = (
                    f"\n## Tests That Must Pass After Fix\n"
                    f"{state['fail_to_pass']}\n\n"
                    f"You should use `run_test` to verify these tests pass after making changes.\n"
                )

            user_msg = (
                f"## Original Issue\n{state['question']}\n\n"
                f"{fail_info}"
                f"## Your Task (Step {node_id})\n{node_desc}\n\n"
                f"## Previously Completed Steps\n"
                f"{prev_summary if prev_summary else '(this is a starting step)'}\n"
            )

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            summary = run_agent_loop(llm_client, model, messages, container)
            print(
                f"  [Done] Node {node_id}: {summary[:150]}{'...' if len(summary) > 150 else ''}"
            )

            return {"node_results": {node_id: summary}, "current_node": node_id}

        return node_fn

    graph = StateGraph(AgentState)

    for node_id, node_desc in nodes.items():
        graph.add_node(node_id, make_agent_node(node_id, node_desc))

    in_edges_map: dict[str, list[str]] = defaultdict(list)
    out_edges_map: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        out_edges_map[src].append(dst)
        in_edges_map[dst].append(src)

    roots = [n for n in nodes if n not in in_edges_map]
    leaves = [n for n in nodes if n not in out_edges_map]

    for root in roots:
        graph.add_edge(START, root)
    for src, dst in edges:
        graph.add_edge(src, dst)
    for leaf in leaves:
        graph.add_edge(leaf, END)

    if not roots and nodes:
        graph.add_edge(START, list(nodes.keys())[0])
    if not leaves and nodes:
        graph.add_edge(list(nodes.keys())[-1], END)

    return graph


# ─── Main Pipeline ────────────────────────────────────────────────────


def _generate_plan_for_instance(
    llm_client: OpenAI,
    model: str,
    instance: dict,
    plan_dir: Path,
) -> bool:
    """Generate (or skip if cached) a DAG plan for a single instance.
    Returns True if a usable plan exists at the end, False otherwise."""
    instance_id = instance["instance_id"]
    question = instance["problem_statement"]
    plan_file = plan_dir / f"{instance_id}.txt"

    if plan_file.exists():
        plan_text = plan_file.read_text().strip()
        if plan_text:
            print(f"  {instance_id}: plan already exists, skipping.")
            return True
        else:
            print(f"  {instance_id}: plan file empty, regenerating.")

    try:
        plan_text = generate_dag(llm_client, model, question)
    except Exception as e:
        print(f"  {instance_id}: FAILED to generate DAG: {e}")
        return False

    if plan_text is None:
        print(f"  {instance_id}: FAILED to generate DAG (empty or None response).")
        return False

    nodes, edges = parse_dag(plan_text)
    issues = validate_dag(nodes, edges)
    if issues:
        print(f"  [Warning] DAG validation issues: {issues}")
        if not nodes:
            print(f"  Falling back to single-node DAG")
            plan_text = "A: Analyze and fix the issue\n"

    plan_file.write_text(plan_text)
    print(f"  {instance_id}: plan saved to {plan_file}")
    return True


def run_pipeline(args):
    llm_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    instances = load_swebench_dataset(
        args.dataset_name_or_path, args.split, args.instance_ids
    )
    total = len(instances)

    # ─── Mode 1: Generate plans only ─────────────────────────────────────
    if getattr(args, "generate_plans_only", False):
        plan_dir = Path(args.plan_dir)
        plan_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'=' * 70}")
        print(f"PLAN GENERATION MODE ({total} instances)")
        print(f"Plans will be saved to: {plan_dir}")
        print(f"{'=' * 70}")

        for idx, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            print(f"\n[{idx + 1}/{total}] {instance_id}")
            _generate_plan_for_instance(llm_client, args.model, instance, plan_dir)
            if args.delay > 0 and idx < total - 1:
                time.sleep(args.delay)

        print(f"\n{'=' * 70}")
        print("All plans generated. You can now run the pipeline without")
        print("--generate-plans-only to execute the agents.")
        print(f"{'=' * 70}")
        return

    # ─── Mode 2: Full agent pipeline ─────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_path
        / f"{args.model}__{Path(args.dataset_name_or_path).name.replace('/', '__')}__{args.split}.jsonl"
    )

    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data["instance_id"])
    if existing_ids:
        print(f"Found {len(existing_ids)} already-completed instances, will skip them.")

    print(f"Output will be written to {output_file}")

    for idx, instance in enumerate(instances):
        instance_id = instance["instance_id"]

        if instance_id in existing_ids:
            print(f"\n[{idx + 1}/{total}] Skipping {instance_id} (already done)")
            continue

        question = instance["problem_statement"]

        fail_to_pass_raw = instance.get("FAIL_TO_PASS", "[]")
        if isinstance(fail_to_pass_raw, str):
            fail_to_pass_list = json.loads(fail_to_pass_raw)
        else:
            fail_to_pass_list = fail_to_pass_raw
        fail_to_pass = (
            "\n".join(f"  - {t}" for t in fail_to_pass_list)
            if fail_to_pass_list
            else ""
        )

        print(f"\n{'=' * 70}")
        print(f"[{idx + 1}/{total}] {instance_id}")
        print(f"  Repo: {instance.get('repo', 'N/A')}")
        print(f"  Issue: {question[:150]}...")
        if fail_to_pass_list:
            print(f"  FAIL_TO_PASS: {len(fail_to_pass_list)} test(s)")
        print(f"{'=' * 70}")

        container = None

        try:
            if args.container:
                print(f"\n[Step 0] Connecting to existing container...")
                container, docker_client = connect_container(args.container)
            else:
                print(
                    f"\n[Step 0] Pulling pre-built Docker image and starting container..."
                )
                container, docker_client = start_container(instance)

            print(f"\n[Step 1] Loading/generating DAG plan...")
            plan_dir = Path(args.plan_dir)
            plan_file = plan_dir / f"{instance_id}.txt"
            plan_text = None

            if plan_file.exists():
                print(f"  Found cached plan: {plan_file}")
                plan_text = plan_file.read_text().strip()
                if not plan_text:
                    print(f"  [Warning] Plan file is empty, will regenerate.")
                    plan_text = None

            if plan_text is None:
                ok = _generate_plan_for_instance(
                    llm_client, args.model, instance, plan_dir
                )
                if ok and plan_file.exists():
                    plan_text = plan_file.read_text().strip()

            if plan_text is not None:
                print(f"[Step 1] DAG:\n{plan_text}")
                nodes, edges = parse_dag(plan_text)
                issues = validate_dag(nodes, edges)
                if issues:
                    print(f"[Warning] DAG validation issues: {issues}")
                    if not nodes:
                        print(f"  Falling back to single-node DAG")
                        nodes, edges = fallback_single_node_dag(question)
            else:
                print(f"[Step 1] No plan available. Using single-node fallback.")
                nodes, edges = fallback_single_node_dag(question)

            print(f"\n[Step 2] Parsed DAG: {len(nodes)} nodes, {len(edges)} edges")
            for nid, desc in nodes.items():
                print(f"  {nid}: {desc}")
            for src, dst in edges:
                print(f"  {src} -> {dst}")

            print(
                f"\n[Step 3] Building LangGraph and running agents inside container..."
            )
            graph = build_langgraph(
                llm_client,
                args.model,
                question,
                fail_to_pass,
                nodes,
                edges,
                container,
            )
            app = graph.compile()

            initial_state: AgentState = {
                "question": question,
                "fail_to_pass": fail_to_pass,
                "node_results": {},
                "current_node": "",
            }

            result = app.invoke(initial_state)

            print(f"\n[Step 4] Extracting patch from container...")
            patch = extract_patch_from_container(
                container, instance.get("base_commit", "")
            )

            if not patch:
                print(
                    f"  [Warning] No patch generated! Agent may not have made any changes."
                )

            print(f"\n[Step 5] Execution results:")
            print(f"{'-' * 70}")
            for nid, desc in nodes.items():
                res = result.get("node_results", {}).get(nid, "[not executed]")
                print(f"\n--- Node {nid}: {desc} ---")
                print(f"{res[:600]}{'...' if len(res) > 600 else ''}")
            print(f"{'-' * 70}")

            full_output = "\n\n".join(
                f"Step {nid}: {desc}\n{result.get('node_results', {}).get(nid, '[not executed]')}"
                for nid, desc in nodes.items()
            )

            pred = {
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": patch,
                "full_output": full_output,
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(pred) + "\n")

        except Exception as e:
            print(f"[{instance_id}] Pipeline error: {e}")
            traceback.print_exc()
            pred = {
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": "",
                "full_output": f"Pipeline error: {e}",
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(pred) + "\n")

        finally:
            if container is not None and not args.keep_container:
                print(f"\n  Cleaning up container...")
                try:
                    container.stop(timeout=10)
                    container.remove(force=True)
                except Exception as e:
                    print(f"  [Warning] Failed to cleanup container: {e}")

        if args.delay > 0 and idx < total - 1:
            time.sleep(args.delay)

    print(f"\nDone. Output written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench + LangGraph Agent Pipeline (Docker-based)"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path to SWE-bench dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        default=None,
        help="Specific instance ids to run. If None, run all instances.",
    )
    parser.add_argument(
        "--model",
        default="glm-5.1",
        help="Model name (default: glm-5.1)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--run-id",
        default="langgraph_agent",
        help="Run ID for container naming",
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Explicit container name/ID to connect to (skips auto-build)",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Keep the container after agent finishes (default: cleanup)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between instances (seconds)",
    )
    parser.add_argument(
        "--plan_dir",
        type=str,
        default="plans",
        help="Directory to cache/load DAG plan files (default: plans/)",
    )
    parser.add_argument(
        "--generate-plans-only",
        action="store_true",
        help="Only generate DAG plans and save them to --plan_dir, then exit. "
        "Useful for pre-generating plans before running the full agent pipeline.",
    )
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY")
    if not args.api_key:
        parser.error("No API key. Use --api-key or set OPENAI_API_KEY env var.")

    if not args.base_url:
        args.base_url = os.environ.get("OPENAI_BASE_URL")
    if not args.base_url:
        parser.error("No base URL. Use --base-url or set OPENAI_BASE_URL env var.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
