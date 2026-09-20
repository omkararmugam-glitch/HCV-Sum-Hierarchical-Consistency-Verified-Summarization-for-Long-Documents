<#
.SYNOPSIS
    Run one or more documents through the HCV-Sum pipeline with the evidence panel and an auto-named JSON.

.DESCRIPTION
    A convenience wrapper around `python -m hcv_sum.cli`. It changes no CLI behaviour, flag or default: it
    only fills in the parts that are the same on every run.

      * activates the project virtual environment if it is not active already
      * adds --evidence, so the final summary, the evidence panel, RUN LIMITS and peak memory are all in
        the terminal (no separate search through a saved JSON)
      * adds --json outputs\<category>\<name>_<timestamp>.json, so no run overwrites an earlier one, where
        <category> is small, medium or large by the document's size on disk, or multidoc for several
        documents (see Get-RunCategory below for the thresholds)
      * prints where that JSON was written, and which category it was filed under

    Document paths come FIRST, then any extra flags, which are passed straight through to the CLI. If you
    pass an output mode yourself (--brief, --evidence, --verbose) the wrapper does not add --evidence, and
    if you pass your own --json it does not add one.

.EXAMPLE
    .\run.ps1 "data\samples\gao-26-108073.pdf"

.EXAMPLE
    .\run.ps1 "data\samples\doc1.pdf" "data\samples\doc2.pdf"

.EXAMPLE
    .\run.ps1 "data\samples\gao-26-108073.pdf" --brief --set contradiction.threshold=0.7
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

# ---------------------------------------------------------------- split "documents first, then flags"
$documents = @()
$extra = @()
$inFlags = $false
foreach ($arg in $Arguments) {
    if (-not $inFlags -and $arg -like '-*') { $inFlags = $true }
    if ($inFlags) { $extra += $arg } else { $documents += $arg }
}

if ($documents.Count -eq 0) {
    Write-Host "usage: .\run.ps1 <document> [more documents ...] [extra CLI flags ...]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  .\run.ps1 `"data\samples\report.pdf`"                     one document"
    Write-Host "  .\run.ps1 `"data\samples\a.pdf`" `"data\samples\b.pdf`"    several documents (multi-document mode)"
    Write-Host "  .\run.ps1 `"data\samples\report.pdf`" --brief            extra flags are passed through"
    Write-Host ""
    Write-Host "Document paths must come before any flags. Supported input: .txt .md .pdf .json .docx"
    exit 2
}

$missing = @($documents | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    Write-Host "error: file not found:" -ForegroundColor Red
    foreach ($path in $missing) { Write-Host "  $path" }
    Write-Host "Give the path to the document as the FIRST argument, flags after it."
    exit 2
}

# ---------------------------------------------------------------- virtual environment
# The primary environment first: %USERPROFILE%\.venvs\hcv-sum (Python 3.11.9) is the one requirements.txt
# pins, it holds every runtime dependency, and it is what every measurement in FINDINGS was made in. It also
# lives outside OneDrive on purpose, because torch is ~2 GB across tens of thousands of files.
# The project-local venv\ (Python 3.14.6) is a secondary forward-compatibility environment: its torch, numpy
# and scipy are newer than the pins and it has no python-docx, so .docx input fails there.
$venvCandidates = @(
    (Join-Path $env:USERPROFILE '.venvs\hcv-sum'),
    (Join-Path $projectRoot 'venv')
)

if ($env:VIRTUAL_ENV) {
    Write-Host "venv: already active ($env:VIRTUAL_ENV)" -ForegroundColor DarkGray
}
else {
    $activated = $false
    foreach ($venv in $venvCandidates) {
        $activateScript = Join-Path $venv 'Scripts\Activate.ps1'
        if (-not (Test-Path -LiteralPath $activateScript)) { continue }
        try {
            . $activateScript
            $activated = $true
            Write-Host "venv: activated $venv" -ForegroundColor DarkGray
            break
        }
        catch {
            # The usual cause is the execution policy refusing to run Activate.ps1, which otherwise
            # surfaces as an opaque "cannot be loaded because running scripts is disabled" error.
            Write-Host "error: could not activate the virtual environment at $venv" -ForegroundColor Red
            Write-Host "  $($_.Exception.Message)"
            Write-Host ""
            Write-Host "If that message mentions the execution policy, allow local scripts for your user:"
            Write-Host "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" -ForegroundColor Yellow
            Write-Host "Then run this script again. To avoid changing the policy, activate the venv by hand:"
            Write-Host "  $venv\Scripts\activate" -ForegroundColor Yellow
            exit 3
        }
    }
    if (-not $activated) {
        Write-Host "error: no virtual environment found to activate." -ForegroundColor Red
        Write-Host "Looked for Scripts\Activate.ps1 in:"
        foreach ($venv in $venvCandidates) { Write-Host "  $venv" }
        Write-Host ""
        Write-Host "Activate yours manually and run this script again, e.g.:"
        Write-Host "  $($venvCandidates[0])\Scripts\activate" -ForegroundColor Yellow
        exit 3
    }
}

# Refuse to run against a system Python: that would install nothing, or silently use the wrong packages.
if (-not $env:VIRTUAL_ENV) {
    Write-Host "error: no virtual environment is active, so this would run the wrong Python." -ForegroundColor Red
    Write-Host "Activate it manually and try again:  $($venvCandidates[0])\Scripts\activate" -ForegroundColor Yellow
    exit 3
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "error: 'python' is not on PATH even though $env:VIRTUAL_ENV is active." -ForegroundColor Red
    Write-Host "The virtual environment looks broken; recreate it (see the README setup section)."
    exit 3
}

# ---------------------------------------------------------------- category + automatic JSON path
# Runs are filed by what produced them: outputs\small, medium, large or multidoc. More than one document is
# always multidoc, whatever the sizes; a single document is bucketed by its size on disk, which can be read
# before the run without opening the file.
#
# Two sets of thresholds, because a byte means different things in different formats: 22 pages of extracted
# prose is 56 KB, while 22 pages of a GAO PDF is several MB of images and embedded fonts. A single cut-off
# cannot serve both -- at one flat 100 KB line, data\samples\gao_excerpt.txt (56 KB, ~22 pages) would be
# filed as 'small' beside the 3 KB samples. In this corpus .txt/.md/.json run ~2.3 KB per page, so 25 KB is
# about 11 pages and 250 KB about 108; the GAO PDFs run ~40 KB per page.
function Get-RunCategory {
    param([string] $Path)

    $bytes = (Get-Item -LiteralPath $Path).Length
    $extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    if ($extension -eq '.pdf' -or $extension -eq '.docx') {
        if ($bytes -lt 100KB) { return 'small' }
        if ($bytes -le 2MB) { return 'medium' }
        return 'large'
    }
    if ($bytes -lt 25KB) { return 'small' }
    if ($bytes -le 250KB) { return 'medium' }
    return 'large'
}

if ($documents.Count -gt 1) { $category = 'multidoc' } else { $category = Get-RunCategory $documents[0] }

$stem = [System.IO.Path]::GetFileNameWithoutExtension($documents[0])
$stem = $stem -replace '[^A-Za-z0-9._-]', '_'
if ($documents.Count -gt 1) { $stem = "${stem}_and_$($documents.Count - 1)_more" }
$outputDir = Join-Path (Join-Path $projectRoot 'outputs') $category
if (-not (Test-Path -LiteralPath $outputDir)) { New-Item -ItemType Directory -Path $outputDir -Force | Out-Null }
$jsonPath = Join-Path $outputDir "${stem}_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"

# ---------------------------------------------------------------- build and run the CLI command
$modeFlags = @('--evidence', '--brief', '--verbose', '-v')
$givenMode = @($extra | Where-Object { $modeFlags -contains $_ })
$givenJson = @($extra | Where-Object { $_ -eq '--json' })

$cliArgs = @('-m', 'hcv_sum.cli') + $documents
if ($givenMode.Count -eq 0) { $cliArgs += '--evidence' }
if ($givenJson.Count -eq 0) { $cliArgs += @('--json', $jsonPath) } else { $jsonPath = $null }
$cliArgs += $extra

Write-Host "run: python $($cliArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& python @cliArgs
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -ne 0) {
    Write-Host "run FAILED (exit code $exitCode); see the error above." -ForegroundColor Red
    exit $exitCode
}
if ($jsonPath) {
    if (Test-Path -LiteralPath $jsonPath) {
        Write-Host "full run saved to: $jsonPath  (category: $category)" -ForegroundColor Green
    }
    else {
        Write-Host "note: the run finished but $jsonPath was not written." -ForegroundColor Yellow
    }
}
else {
    Write-Host "JSON path was given with your own --json; nothing auto-named was written." -ForegroundColor DarkGray
}
exit 0
