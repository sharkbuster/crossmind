# 一键把 CrossMind 推到 GitHub
#
# 首次使用：
#   1) 先到 https://github.com/new 建一个空仓库，名字填 crossmind
#      （不要勾选 README / .gitignore / license，否则首推会被拒）
#   2) 在项目根目录执行：  pwsh -File scripts/push_github.ps1
#
# 如果你的用户名不是 sharkbuster，加参数：
#   pwsh -File scripts/push_github.ps1 -User 你的用户名

param(
    [string]$User = "sharkbuster",
    [string]$Repo = "crossmind",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$url = "https://github.com/$User/$Repo.git"
Write-Host "==> 目标仓库: $url"

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

Write-Host "==> 检查远端是否可达（仓库不存在时会失败，请先在 GitHub 上创建）"
git ls-remote --exit-code --heads origin *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> 远端为空或不可达，继续尝试推送（首次推送空仓库是正常的）"
}

Write-Host "==> 推送 $Branch"
git push -u origin $Branch

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "推送完成：https://github.com/$User/$Repo"
    Write-Host "建议到仓库 Settings 里补：Description、Topics（rag, langgraph, fastapi, multi-agent）"
} else {
    Write-Host ""
    Write-Host "推送失败。常见原因："
    Write-Host "  - 远端仓库还没创建，或名字不是 $Repo"
    Write-Host "  - 未登录：Windows 凭据管理器里没有 GitHub 账号，执行一次 git push 会弹登录框"
    Write-Host "  - 远端已有提交（建仓库时勾了 README），可先执行：git pull --rebase origin $Branch"
}
