#!/usr/bin/env python3
# 中文功能说明：Crawler Strategy 远端部署同步脚本，将训练项目上传到 cu05。

"""Deploy the crawler strategy training package to cu05 through mu01."""

from __future__ import annotations

import argparse
import os
import posixpath
import socket
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko


EXCLUDE_PARTS = {
    ".venv",
    ".git",
    "__pycache__",
    ".cache",
    "outputs",
    "logs",
}


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def should_exclude(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts) or any(part.endswith(".pyc") for part in path.parts)


def make_package(model_root: Path) -> Path:
    package_path = Path(tempfile.gettempdir()) / f"{model_root.name}_{int(time.time())}.tar.gz"
    with tarfile.open(package_path, "w:gz") as tar:
        for path in model_root.rglob("*"):
            rel = path.relative_to(model_root.parent)
            if should_exclude(rel):
                continue
            tar.add(path, arcname=rel)
    return package_path


def connect_ssh(host: str, user: str, password: str, sock: socket.socket | None = None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        username=user,
        password=password,
        sock=sock,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def run(client: paramiko.SSHClient, command: str, check: bool = True) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if check and code != 0:
        raise RuntimeError(f"Command failed ({code}): {command}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return code, out, err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-parent", default="/data/model_train")
    parser.add_argument("--start", action="store_true", help="Start training in the background after upload.")
    parser.add_argument(
        "--config-path",
        default="configs/crawler_qwen2p5_3b_lora.yaml",
        help="Training config used when --start is set.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    model_root = project_dir.parent
    package_path = make_package(model_root)
    remote_parent = args.remote_parent.rstrip("/")
    remote_root = posixpath.join(remote_parent, model_root.name)
    remote_package = posixpath.join(remote_parent, package_path.name)

    jump_host = require_env("JUMP_HOST")
    jump_user = require_env("JUMP_USER")
    jump_password = require_env("JUMP_PASSWORD")
    target_host = require_env("TARGET_HOST")
    target_user = require_env("TARGET_USER")
    target_password = require_env("TARGET_PASSWORD")

    print(f"package={package_path}")
    print(f"remote_root={remote_root}")

    jump = connect_ssh(jump_host, jump_user, jump_password)
    try:
        jump_transport = jump.get_transport()
        if jump_transport is None:
            raise RuntimeError("Jump SSH transport is not available.")
        channel = jump_transport.open_channel("direct-tcpip", (target_host, 22), ("127.0.0.1", 0))
        target = connect_ssh(target_host, target_user, target_password, sock=channel)
        try:
            run(target, f"mkdir -p {remote_parent!r}")
            with target.open_sftp() as sftp:
                sftp.put(str(package_path), remote_package)
            run(target, f"tar -xzf {remote_package!r} -C {remote_parent!r}")
            run(target, f"chmod +x {remote_root!r}/crawler_strategy_project/scripts/*.sh")
            code, out, err = run(
                target,
                "cd "
                + f"{remote_root!r}/crawler_strategy_project"
                + " && find . -maxdepth 3 -type f | sort | sed -n '1,80p'",
            )
            del code
            print(out)
            if err:
                print(err)

            if args.start:
                command = (
                    f"cd {remote_root!r}/crawler_strategy_project "
                    f"&& CONFIG_PATH={args.config_path!r} bash scripts/start_training_background.sh"
                )
                _, out, err = run(target, command)
                print(out)
                if err:
                    print(err)
        finally:
            target.close()
    finally:
        jump.close()
        try:
            package_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
