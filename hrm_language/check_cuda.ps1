# Check CUDA Installation Script
Write-Host "Checking CUDA installation..." -ForegroundColor Blue

# Check NVIDIA driver
try {
    $nvidiaOutput = nvidia-smi
    Write-Host "NVIDIA driver found:" -ForegroundColor Green
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
} catch {
    Write-Host "NVIDIA driver not found or not working" -ForegroundColor Red
}

Write-Host "" -ForegroundColor White

# Check CUDA toolkit paths
$cudaPaths = @(
    "${env:CUDA_PATH}",
    "${env:CUDA_HOME}",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0", 
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
)

$foundCuda = $false
foreach ($path in $cudaPaths) {
    if ($path -and (Test-Path "$path\bin\nvcc.exe")) {
        Write-Host "CUDA Toolkit found at: $path" -ForegroundColor Green
        & "$path\bin\nvcc.exe" --version
        $foundCuda = $true
        break
    }
}

if (-not $foundCuda) {
    Write-Host "CUDA Toolkit not found" -ForegroundColor Yellow
    Write-Host "FlashAttention requires CUDA Toolkit (not just drivers)" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor White
    Write-Host "To install CUDA Toolkit:" -ForegroundColor Blue
    Write-Host "1. Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    Write-Host "2. Choose: Windows > x86_64 > 10/11 > exe (local)" -ForegroundColor White
    Write-Host "3. Install with default options" -ForegroundColor White
    Write-Host "4. Restart PowerShell" -ForegroundColor White
} else {
    Write-Host "" -ForegroundColor White
    Write-Host "CUDA is properly installed! FlashAttention should work." -ForegroundColor Green
} 