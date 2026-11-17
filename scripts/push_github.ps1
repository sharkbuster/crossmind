# 一键把 CrossMind 推到 GitHub
#
# 首次使用：
#   1) 先到 https://github.com/new 建一个空仓库，名字填 crossmind
#      （不要勾选 README / .gitignore / license，否则首推会被拒）
#   2) 在项目根目录执行：  pwsh -File scripts/push_github.ps1
#
# 默认走 SSH（本机已验证可用；HTTPS 到 github.com 在部分网络会被 SNI 拦截）。
# 如果你的用户名不是 sharkbuster：  pwsh -File scripts/push_github.ps1 -User 你的用户名
# 如果只配置了 HTTPS 凭据：        pwsh -File scripts/push_github.ps1 -Protocol Https

param(
    [string]$User = "sharkbuster",
    [string]$Repo = "crossmind",
    [string]$Branch = "main",
    [ValidateSet("Ssh", "Https")]
    [string]$Protocol = "Ssh"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$url = if ($Protocol -eq "Ssh") {
    "git@github.com:$User/$Repo.git"
} else {
    "https://github.com/$User/$Repo.git"
}

Write-Host "==> 目标仓库: $url"
if ($Protocol -eq "Ssh") {
    Write-Host "==> 校验 SSH 身份（未添加公钥会提示 Permission denied）"
    $auth = ssh -o BatchMode=yes -o ConnectTimeout=10 -T "git@github.com" 2>&1
    if ($auth -notmatch "successfully authenticated") {
        Write-Host $auth
        throw "SSH 认证未通过：请把 ~/.ssh/id_ed25519.pub 加到 GitHub → Settings → SSH keys"
    }
}

if (Test-Path ".git") {
    $existing = git remote get-url origin 2>$null
    if ($LASTEXITCODE -eq 0 -and $existing) {
        git remote set-url origin $url
    } else {
        git remote add origin $url
    }
} else {
    throw "当前目录不是 git 仓库：$root"
}

Write-Host "==> 推送 $Branch"
git push -u origin $Branch

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "推送完成：https://github.com/$User/$Repo"
    Write-Host "建议到仓库 About 里补 Description 与 Topics（rag, langgraph, fastapi, multi-agent）"
} else {
    Write-Host ""
    Write-Host "推送失败。常见原因："
    Write-Host "  - 远端仓库还没创建，或名字不是 $Repo"
    Write-Host "  - HTTPS 被网络拦截（SNI），改用默认的 SSH 协议"
    Write-Host "  - 远端已有提交（建仓库时勾了 README），先执行：git pull --rebase origin $Branch"
}
