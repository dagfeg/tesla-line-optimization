"""
deploy_to_github.py
将本项目自动推送到 GitHub 公开仓库（无需本地安装 git）。

使用方法：
    python deploy_to_github.py --token YOUR_GITHUB_TOKEN --repo tesla-line-optimization

需要 GitHub Personal Access Token（classic），至少勾选 public_repo 权限。
推送完成后，打开 https://share.streamlit.io 登录 GitHub 即可一键部署。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request


API_BASE = "https://api.github.com"


def api_request(method: str, url: str, token: str, data: dict | None = None):
    """发送 GitHub API 请求并返回 JSON"""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "streamlit-deploy-script",
    }
    body = json.dumps(data).encode("utf-8") if data else None
    if body:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        raise RuntimeError(f"GitHub API error ({e.code}): {err}") from e


def get_username(token: str) -> str:
    """获取 Token 对应的 GitHub 用户名"""
    user = api_request("GET", f"{API_BASE}/user", token)
    return user["login"]


def create_repo(token: str, owner: str, repo: str) -> dict:
    """创建公开仓库；若已存在则返回仓库信息"""
    url = f"{API_BASE}/user/repos"
    try:
        return api_request("POST", url, token, {
            "name": repo,
            "description": "基于瓶颈理论与离散事件仿真的汽车生产线优化平台",
            "private": False,
            "auto_init": False,
        })
    except RuntimeError as e:
        if "422" in str(e):
            # 仓库已存在，直接获取
            return api_request("GET", f"{API_BASE}/repos/{owner}/{repo}", token)
        raise


def should_upload(rel_path: str) -> bool:
    """过滤不需要上传的文件"""
    skip_parts = {"__pycache__", ".git", ".venv", "venv", "env", ".cloudflared"}
    parts = set(rel_path.replace("\\", "/").split("/"))
    if parts & skip_parts:
        return False
    if rel_path.endswith((".pyc", ".pyo", ".pyd", ".log", ".DS_Store")):
        return False
    if rel_path in {"cloudflared.exe"}:
        return False
    return True


def upload_file(token: str, owner: str, repo: str, path: str, local_path: str):
    """上传单个文件到 GitHub"""
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"
    try:
        return api_request("PUT", url, token, {
            "message": f"add {path}",
            "content": content_b64,
        })
    except RuntimeError as e:
        if "422" in str(e):
            # 文件已存在，获取 sha 后更新
            existing = api_request("GET", url, token)
            sha = existing["sha"]
            return api_request("PUT", url, token, {
                "message": f"update {path}",
                "content": content_b64,
                "sha": sha,
            })
        raise


def main():
    parser = argparse.ArgumentParser(description="Deploy project to GitHub")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--repo", required=True, help="目标仓库名，例如 tesla-line-optimization")
    args = parser.parse_args()

    print("验证 GitHub Token...")
    owner = get_username(args.token)
    print(f"用户名: {owner}")

    print(f"创建/获取仓库: {args.repo}...")
    repo_info = create_repo(args.token, owner, args.repo)
    print(f"仓库地址: {repo_info['html_url']}")

    print("扫描本地文件...")
    root = os.path.dirname(os.path.abspath(__file__))
    uploaded = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            local_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(local_path, root).replace("\\", "/")
            if not should_upload(rel_path):
                continue
            print(f"  上传 {rel_path} ...")
            upload_file(args.token, owner, args.repo, rel_path, local_path)
            uploaded += 1

    print(f"\n成功上传 {uploaded} 个文件。")
    print(f"仓库首页: {repo_info['html_url']}")
    print("下一步：")
    print("  1. 打开 https://share.streamlit.io")
    print("  2. 用同一 GitHub 账号登录")
    print(f"  3. 点击 New app，选择仓库 {owner}/{args.repo}，主文件路径填 app.py，然后 Deploy")


if __name__ == "__main__":
    main()
