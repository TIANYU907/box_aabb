param(
    [ValidateSet("conda", "pip")]
    [string]$Mode = "conda",

    [string]$EnvName = "box-rrt",

    [string]$PythonVersion = "3.10",

    [switch]$CreateEnv,

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

$V2Root = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "[setup] mode=$Mode" -ForegroundColor Green
Write-Host "[setup] v2_root=$V2Root" -ForegroundColor Green

if ($Mode -eq "conda") {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "conda was not found. Install Miniconda/Anaconda first, or use -Mode pip."
    }

    if ($CreateEnv) {
        Invoke-Step "conda create -n $EnvName python=$PythonVersion -y"
    }

    Invoke-Step "conda install -n $EnvName -c conda-forge numpy scipy matplotlib pytest ompl drake -y"
    Invoke-Step "conda run -n $EnvName python -m pip install --upgrade pip"
    Invoke-Step "conda run -n $EnvName python -m pip install -e `"$V2Root`""

    if ($RunSmokeTest) {
        Invoke-Step "conda run -n $EnvName python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --help"
    }

    Write-Host "[done] conda dependency setup completed: $EnvName" -ForegroundColor Green
    return
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found. Install Python first, or use -Mode conda."
}

Invoke-Step "python -m pip install --upgrade pip"
Invoke-Step "python -m pip install numpy scipy matplotlib pytest"
Invoke-Step "python -m pip install ompl drake"
Invoke-Step "python -m pip install -e `"$V2Root`""

if ($RunSmokeTest) {
    Invoke-Step "python -m v2.benchmarks.planner.bench_rrt_vs_marcucci --help"
}

Write-Host "[done] pip dependency setup completed (current Python environment)" -ForegroundColor Green
