param(
    [ValidateSet("conda", "pip")]
    [string]$Mode = "conda",

    [string]$EnvName = "box-rrt",

    [string]$PythonVersion = "3.10",

    [switch]$CreateEnv,

    [switch]$SkipSetup,

    [string]$Robot = "2dof_planar",

    [int]$Trials = 8,

    [int]$Seed = 42,

    [double]$OmplTimeout = 1.5,

    [switch]$RunSmokeTest
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )
    Write-Host "> $Command" -ForegroundColor Cyan
    Invoke-Expression $Command
}

$ScriptsRoot = Resolve-Path $PSScriptRoot
$V2Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$SetupScript = Join-Path $ScriptsRoot "setup_benchmark_env.ps1"

Write-Host "[oneclick] mode=$Mode" -ForegroundColor Green
Write-Host "[oneclick] v2_root=$V2Root" -ForegroundColor Green

if (-not $SkipSetup) {
    if (-not (Test-Path $SetupScript)) {
        throw "setup script not found: $SetupScript"
    }

    $setupArgs = @(
        "-Mode", $Mode,
        "-EnvName", $EnvName,
        "-PythonVersion", $PythonVersion
    )
    if ($CreateEnv) {
        $setupArgs += "-CreateEnv"
    }
    if ($RunSmokeTest) {
        $setupArgs += "-RunSmokeTest"
    }

    Write-Host "[oneclick] running setup script..." -ForegroundColor Yellow
    & $SetupScript @setupArgs
} else {
    Write-Host "[oneclick] setup skipped by -SkipSetup" -ForegroundColor Yellow
}

$benchCmd = "-m v2.benchmarks.planner.bench_rrt_vs_marcucci --robot $Robot --trials $Trials --seed $Seed --ompl-timeout $OmplTimeout"

if ($Mode -eq "conda") {
    Invoke-Step "conda run -n $EnvName python $benchCmd"
} else {
    Invoke-Step "python $benchCmd"
}

$BenchOutRoot = Join-Path $V2Root "output\benchmarks"
if (-not (Test-Path $BenchOutRoot)) {
    throw "benchmark output root not found: $BenchOutRoot"
}

$latestDir = Get-ChildItem -Path $BenchOutRoot -Directory -Filter "planner_rrt_vs_marcucci_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $latestDir) {
    throw "no benchmark output directory found matching planner_rrt_vs_marcucci_*"
}

$summaryPath = Join-Path $latestDir.FullName "summary.md"
if (-not (Test-Path $summaryPath)) {
    throw "summary.md not found in latest benchmark output: $($latestDir.FullName)"
}

$latestAliasPath = Join-Path $BenchOutRoot "planner_rrt_vs_marcucci_latest_summary.md"
$sourceText = Get-Content -Path $summaryPath -Raw -Encoding UTF8
$header = @(
    "# Latest Planner Benchmark Summary",
    "",
    "- generated_at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "- source_dir: $($latestDir.FullName)",
    "- source_summary: $summaryPath",
    ""
) -join "`n"

Set-Content -Path $latestAliasPath -Value ($header + $sourceText) -Encoding UTF8

Write-Host "[done] latest benchmark dir: $($latestDir.FullName)" -ForegroundColor Green
Write-Host "[done] source summary: $summaryPath" -ForegroundColor Green
Write-Host "[done] latest alias: $latestAliasPath" -ForegroundColor Green
