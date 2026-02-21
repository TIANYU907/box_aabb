# Build script for IROS paper
# Requires: pdflatex, bibtex (TeX Live / MiKTeX)
# Also requires: IEEEtran.cls (download from ieee.org or CTAN)

param(
    [switch]$Clean,
    [switch]$Once
)

Push-Location $PSScriptRoot

if ($Clean) {
    Remove-Item -Force -ErrorAction SilentlyContinue main.aux, main.bbl, main.blg, main.log, main.out, main.pdf, main.synctex.gz
    Write-Host "Cleaned build artifacts."
    Pop-Location
    return
}

# Check IEEEtran.cls
if (-not (Test-Path "IEEEtran.cls")) {
    Write-Host "ERROR: IEEEtran.cls not found."
    Write-Host "Download from: https://www.ieee.org/conferences/publishing/templates.html"
    Write-Host "Or: https://ctan.org/pkg/ieeetran"
    Write-Host "Place IEEEtran.cls in this directory."
    Pop-Location
    exit 1
}

# Check pdflatex
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: pdflatex not found. Install TeX Live or MiKTeX."
    Pop-Location
    exit 1
}

Write-Host "=== Pass 1: pdflatex ==="
pdflatex -interaction=nonstopmode main.tex
if (-not $Once) {
    Write-Host "=== BibTeX ==="
    bibtex main
    Write-Host "=== Pass 2: pdflatex ==="
    pdflatex -interaction=nonstopmode main.tex
    Write-Host "=== Pass 3: pdflatex ==="
    pdflatex -interaction=nonstopmode main.tex
}

if (Test-Path "main.pdf") {
    Write-Host "=== SUCCESS: main.pdf generated ==="
} else {
    Write-Host "=== FAILED: main.pdf not generated ==="
}

Pop-Location
