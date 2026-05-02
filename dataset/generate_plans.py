#!/usr/bin/env python3
"""
Generate DAG-format plans for each question in the 'questions' file
using an OpenAI-compatible API.
"""

import os
import time
import argparse
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """\
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
6. Output NOTHING else besides the DAG — no markdown, no explanation, no preamble.
"""


def load_questions(path: str) -> list[str]:
    questions: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Strip "N: " line-number prefix from the questions file format
            if line and line[0].isdigit():
                colon_idx = line.find(":")
                if colon_idx != -1:
                    line = line[colon_idx + 1 :].lstrip()
            if line.strip():
                questions.append(line.strip())
    return questions


def generate_dag(client: OpenAI, model: str, question: str) -> str:
    user_content = f"Here is the issue:\n\n{question}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def validate_dag(dag_text: str) -> list[str]:
    issues: list[str] = []
    lines = dag_text.strip().split("\n")

    node_section = True
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []

    for line in lines:
        line = line.strip()
        if not line:
            if node_section and nodes:
                node_section = False
            continue

        if node_section:
            if ":" not in line:
                issues.append(f"Invalid node line (no colon): {line}")
                continue
            letter, _, desc = line.partition(":")
            letter = letter.strip()
            desc = desc.strip()
            if len(letter) != 1 or not letter.isalpha():
                issues.append(f"Invalid node identifier: {letter}")
                continue
            nodes[letter] = desc
        else:
            if "->" not in line:
                issues.append(f"Invalid edge line (no arrow): {line}")
                continue
            parts = [p.strip() for p in line.split("->")]
            if len(parts) != 2:
                issues.append(f"Invalid edge format: {line}")
                continue
            src, dst = parts
            if src not in nodes:
                issues.append(f"Edge references undefined node: {src}")
            if dst not in nodes:
                issues.append(f"Edge references undefined node: {dst}")
            edges.append((src, dst))

    if not nodes:
        issues.append("No nodes defined")
    if not edges and len(nodes) > 1:
        issues.append("Multiple nodes but no edges defined")

    # 3-color DFS cycle detection
    if edges and nodes:
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for src, dst in edges:
            if src in adj and dst in adj:
                adj[src].append(dst)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}

        def has_cycle(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and has_cycle(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in nodes:
            if color[node] == WHITE:
                if has_cycle(node):
                    issues.append("Graph contains a cycle")
                    break

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Generate DAG plans for issue questions"
    )
    parser.add_argument(
        "--questions-file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions"),
        help="Path to the questions file",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans"),
        help="Directory to write plan files",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (overrides OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this question index (0-based)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Stop processing at this question index (exclusive, 0-based)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retries on API failure",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls to avoid rate limiting",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    questions = load_questions(args.questions_file)
    print(f"Loaded {len(questions)} questions from {args.questions_file}")

    start = args.start_index
    end = args.end_index if args.end_index is not None else len(questions)
    questions_slice = questions[start:end]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, question in enumerate(questions_slice):
        global_idx = start + i
        output_path = output_dir / f"plan_{global_idx:03d}.dag"
        if output_path.exists():
            print(f"[{global_idx}] Plan already exists at {output_path}, skipping.")
            continue

        print(f"[{global_idx}] Generating plan...")
        dag_text = None
        for attempt in range(args.retry):
            try:
                dag_text = generate_dag(client, args.model, question)
                break
            except Exception as e:
                print(f"  Attempt {attempt + 1}/{args.retry} failed: {e}")
                if attempt < args.retry - 1:
                    time.sleep(2**attempt)

        if dag_text is None:
            print(f"[{global_idx}] FAILED after {args.retry} attempts. Skipping.")
            continue

        validation_issues = validate_dag(dag_text)
        if validation_issues:
            print(f"[{global_idx}] Validation warnings:")
            for issue in validation_issues:
                print(f"  - {issue}")
            print(f"  Saving anyway, but the DAG may need manual review.")

        output_path.write_text(dag_text, encoding="utf-8")
        print(f"[{global_idx}] Plan written to {output_path}")

        if args.delay > 0 and i < len(questions_slice) - 1:
            time.sleep(args.delay)

    print("Done.")


if __name__ == "__main__":
    main()
