from __future__ import annotations

import argparse
import posixpath
import shlex
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko


EXCLUDED_DIRS = {".git", "__pycache__", ".venv", "outputs"}
EXCLUDED_TOP_LEVEL = {"data"}


def make_source_tar(source_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in source_root.rglob("*"):
            rel = path.relative_to(source_root)
            parts = set(rel.parts)
            if parts & EXCLUDED_DIRS:
                continue
            if rel.parts and rel.parts[0] in EXCLUDED_TOP_LEVEL:
                continue
            tar.add(path, arcname=Path(source_root.name) / rel, recursive=False)


def make_pasa_subset_tar(pasa_data_dir: Path, archive_path: Path) -> None:
    include_paths = [
        pasa_data_dir / "sft_selector",
        pasa_data_dir / "AutoScholarQuery",
        pasa_data_dir / "RealScholarQuery",
        pasa_data_dir / "paper_database" / "id2paper.json",
    ]
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in include_paths:
            if not path.exists():
                continue
            rel = path.relative_to(pasa_data_dir)
            tar.add(path, arcname=Path("pasa") / "data" / rel)


def ssh_client(host: str, username: str, password: str, *, sock=None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=username,
        password=password,
        sock=sock,
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def run(client: paramiko.SSHClient, command: str, *, timeout: int | None = None) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    del stdin
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return code, out, err


def upload(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    size = local_path.stat().st_size
    print(f"upload {local_path.name} -> {remote_path} ({size / 1024 / 1024:.1f} MiB)")
    sftp.put(str(local_path), remote_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload selector reranker source/data to cu05 and start training.")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pasa-data-dir", type=Path, default=Path(__file__).resolve().parents[4] / "数据集" / "pasa" / "data")
    parser.add_argument("--remote-root", default="/home/model_train")
    parser.add_argument("--venv-dir", default="/home/model_train/py-train2")
    parser.add_argument("--jump-host", default="10.99.24.181")
    parser.add_argument("--jump-user", required=True)
    parser.add_argument("--jump-password", required=True)
    parser.add_argument("--target-host", default="11.11.11.5")
    parser.add_argument("--target-user", default="root")
    parser.add_argument("--target-password", required=True)
    parser.add_argument("--no-start", action="store_true", help="Only upload/extract, do not start training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    pasa_data_dir = args.pasa_data_dir.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not (pasa_data_dir / "sft_selector" / "train.jsonl").exists():
        raise FileNotFoundError(pasa_data_dir / "sft_selector" / "train.jsonl")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        source_tar = tmp / "selector_reranker_model.tar.gz"
        pasa_tar = tmp / "pasa_training_subset.tar.gz"
        make_source_tar(source_root, source_tar)
        make_pasa_subset_tar(pasa_data_dir, pasa_tar)

        print(f"connect jump {args.jump_user}@{args.jump_host}")
        jump = ssh_client(args.jump_host, args.jump_user, args.jump_password)
        try:
            transport = jump.get_transport()
            if transport is None:
                raise RuntimeError("Jump host transport is not available")
            channel = transport.open_channel(
                "direct-tcpip",
                (args.target_host, 22),
                ("127.0.0.1", 0),
            )
            print(f"connect target {args.target_user}@{args.target_host} via jump")
            target = ssh_client(args.target_host, args.target_user, args.target_password, sock=channel)
            try:
                code, out, err = run(target, "hostname && python3 --version 2>/dev/null || python --version")
                print(out.strip())
                if code != 0:
                    raise RuntimeError(err)

                remote_root = args.remote_root.rstrip("/")
                remote_source_tar = posixpath.join(remote_root, source_tar.name)
                remote_pasa_tar = posixpath.join(remote_root, pasa_tar.name)
                run(target, f"mkdir -p {shlex.quote(remote_root)}")
                sftp = target.open_sftp()
                try:
                    upload(sftp, source_tar, remote_source_tar)
                    upload(sftp, pasa_tar, remote_pasa_tar)
                finally:
                    sftp.close()

                extract_cmd = " && ".join(
                    [
                        f"cd {shlex.quote(remote_root)}",
                        "rm -rf selector_reranker_model pasa",
                        f"tar -xzf {shlex.quote(remote_source_tar)}",
                        f"tar -xzf {shlex.quote(remote_pasa_tar)}",
                        "chmod +x selector_reranker_model/scripts/train_cu05.sh",
                        "find selector_reranker_model -maxdepth 2 -type f | sort | head -30",
                    ]
                )
                code, out, err = run(target, extract_cmd, timeout=600)
                print(out)
                if code != 0:
                    raise RuntimeError(err)

                if args.no_start:
                    print("uploaded and extracted; training not started because --no-start was set")
                    return

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_path = posixpath.join(remote_root, "selector_reranker_model", "logs", f"train_{timestamp}.log")
                train_cmd = (
                    f"cd {shlex.quote(posixpath.join(remote_root, 'selector_reranker_model'))} && "
                    "mkdir -p logs && "
                    f"nohup env VENV_DIR={shlex.quote(args.venv_dir)} bash scripts/train_cu05.sh "
                    f"{shlex.quote(posixpath.join(remote_root, 'selector_reranker_model'))} "
                    f"{shlex.quote(posixpath.join(remote_root, 'pasa', 'data'))} "
                    f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
                )
                code, out, err = run(target, train_cmd)
                if code != 0:
                    raise RuntimeError(err)
                print(f"training_pid={out.strip()}")
                print(f"log_path={log_path}")
            finally:
                target.close()
        finally:
            jump.close()


if __name__ == "__main__":
    main()
