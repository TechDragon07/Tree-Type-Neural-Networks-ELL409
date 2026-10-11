param(
    [string]$Tectonic = "tectonic"
)

$ErrorActionPreference = "Stop"

python scripts/summarize_results.py
Push-Location report
try {
    & $Tectonic main.tex --outdir .
    Copy-Item main.pdf corrective_neural_tree_report.pdf -Force
}
finally {
    Pop-Location
}

Write-Output "Built report/corrective_neural_tree_report.pdf"
