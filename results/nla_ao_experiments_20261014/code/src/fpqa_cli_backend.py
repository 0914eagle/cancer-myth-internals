"""Matched experiment messages over isolated Codex/Claude CLI sessions.

Provider-added context and generation defaults are not asserted to be identical.
No automatic fallback, format repair, or retries: transport failure stops the run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from urllib.request import Request, urlopen

COMMON = "Respond in English using only your existing knowledge. Do not use tools, browse, or access files. Return only the requested output."


def envelope(messages):
    """Same system/user bytes for every backend; preserve multi-turn examples as text."""
    if any(m["role"] not in {"system", "user", "assistant"} for m in messages):
        raise ValueError("Unsupported message role")
    system = COMMON + "\n\n" + "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    if len(rest) == 1 and rest[0]["role"] == "user":
        return system, rest[0]["content"]
    return system, ("Complete the final user request in this conversation. Earlier assistant messages are examples.\n"
                    + json.dumps(rest, ensure_ascii=False, separators=(",", ":")))


def run_process(command, prompt, cwd, timeout):
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=cwd, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except BaseException:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    if proc.returncode:
        raise RuntimeError(f"CLI exited {proc.returncode}: {(stderr or stdout)[-1200:]}")
    return stdout, stderr


def parse_claude(stdout, model):
    data = json.loads(stdout)
    if data.get("is_error") or data.get("subtype") != "success":
        raise ValueError(f"Claude failed: {str(data.get('result'))[:500]}")
    if data.get("stop_reason") != "end_turn":
        raise ValueError("Claude output incomplete")
    if data.get("permission_denials") or data.get("num_turns") != 1:
        raise ValueError("Unexpected tool attempt or multi-turn execution")
    usage = data.get("modelUsage", {})
    if set(usage) != {model}:
        raise ValueError(f"Unexpected served model(s): {list(usage)}; expected {model}")
    server = data.get("usage", {}).get("server_tool_use", {})
    if any(server.values()) or data.get("subagent_stats", {}).get("spawned", 0):
        raise ValueError("Unexpected server tool or agent")
    if not data.get("result", "").strip():
        raise ValueError("Empty response")
    return data["result"].strip(), data


def parse_codex(stdout, final):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if any(e.get("type") in {"error", "turn.failed"} for e in events):
        raise ValueError("Codex turn failed")
    if sum(e.get("type") == "turn.completed" for e in events) != 1:
        raise ValueError("Codex did not complete exactly one turn")
    for event in events:
        item = event.get("item")
        if item and item.get("type") not in {"agent_message", "reasoning"}:
            raise ValueError(f"Unexpected Codex tool item: {item.get('type')}")
    answers = [e["item"].get("text", "").strip() for e in events
               if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
    if not final.strip() or not answers or answers[-1] != final.strip():
        raise ValueError("Missing or inconsistent final output")
    return final.strip(), events


class CLIBackend:
    def __init__(self, config):
        if config["backend"] not in {"codex", "claude", "local_http"}:
            raise ValueError("Unsupported experiment backend")
        self.config = config

    def __call__(self, messages):
        c = self.config
        system, user = envelope(messages)
        if c["backend"] == "local_http":
            return self.local_http(system, user)
        with tempfile.TemporaryDirectory(prefix="fpqa-call-") as folder:
            root = Path(folder)
            if c["backend"] == "claude":
                cmd = ["claude", "-p", "--model", c["model"], "--effort", c["effort"],
                       "--safe-mode", "--tools", "", "--disable-slash-commands",
                       "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                       "--no-session-persistence", "--system-prompt", system, "--output-format", "json"]
            else:
                instructions = root / "instructions.txt"
                instructions.write_text(system)
                cmd = ["codex", "exec", "--ignore-user-config", "--skip-git-repo-check",
                       "--ephemeral", "--sandbox", "read-only", "--model", c["model"],
                       "--json", "--output-last-message", str(root / "final.txt")]
                options = {"model_instructions_file": str(instructions),
                           "model_reasoning_effort": c["effort"], "web_search": "disabled",
                           "project_doc_max_bytes": 0}
                for name in ("shell_tool", "apps", "plugins", "multi_agent", "browser_use",
                             "image_generation", "computer_use", "sleep_tool", "view_image",
                             "hooks", "skill_search", "memories"):
                    options[f"features.{name}"] = False
                for key, value in options.items():
                    cmd.extend(["-c", f"{key}={json.dumps(value)}"])
                cmd.append("-")
            stdout, stderr = run_process(cmd, user, folder, c["timeout_seconds"])
            if c["backend"] == "claude":
                answer, raw = parse_claude(stdout, c["model"])
                model_source = "modelUsage returned by Claude CLI"
            else:
                answer, raw = parse_codex(stdout, (root / "final.txt").read_text())
                model_source = "explicit --model; JSON events do not independently expose served model"
            version = subprocess.run([c["backend"], "--version"], capture_output=True, text=True, check=True).stdout.strip()
            return answer, {"backend": c["backend"], "model_requested": c["model"],
                            "model_identity_source": model_source, "cli_version": version,
                            "system_text": system, "user_text": user,
                            "raw": raw, "stderr": stderr}

    def local_http(self, system, user):
        """Prepared local-model bridge. Server is started separately, never by this runner."""
        c = self.config
        base = os.environ[c["base_url_env"]].rstrip("/")
        if not (base.startswith("http://127.0.0.1:") or base.startswith("http://localhost:")):
            raise ValueError("Local model endpoint must be loopback (use an SSH tunnel for another host)")
        body = {"model": c["model"], "messages": [{"role": "system", "content": system},
                {"role": "user", "content": user}], "temperature": c["temperature"],
                "max_tokens": c["max_tokens"], "stream": False, **c.get("extra_body", {})}
        req = Request(base + "/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=c["timeout_seconds"]) as response:
            data = json.load(response)
        choice = data["choices"][0]
        if data.get("model") != c["model"] or choice["finish_reason"] != "stop" or choice["message"].get("tool_calls"):
            raise ValueError("Wrong model, incomplete output, or tool call from local endpoint")
        answer = choice["message"]["content"]
        if not answer or not answer.strip():
            raise ValueError("Empty local response")
        return answer.strip(), {"system_text": system, "user_text": user, "raw": data,
                                "model_identity_source": "local server response; checkpoint must be verified at deployment"}
