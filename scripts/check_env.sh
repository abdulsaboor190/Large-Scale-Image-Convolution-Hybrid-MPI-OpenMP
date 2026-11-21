#!/usr/bin/env bash
# Phase 0 environment check. Run: bash scripts/check_env.sh
ok=1
chk() { # name, command
  if command -v "$2" >/dev/null 2>&1; then printf '  [ok]   %-10s %s\n' "$1" "$($2 --version 2>&1 | head -1)"; else printf '  [MISS] %-10s not found\n' "$1"; ok=0; fi
}
echo "Toolchain"
chk gcc gcc
chk mpicc mpicc
chk mpirun mpirun
chk make make
chk python3 python3
echo "Python packages"
for p in numpy PIL matplotlib; do
  if python3 -c "import $p" 2>/dev/null; then echo "  [ok]   $p"; else echo "  [MISS] $p"; ok=0; fi
done
echo "Machine"
cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)
echo "  logical cores: $cores"
command -v free >/dev/null && free -g | awk '/Mem:/ {print "  RAM: " $2 " GB total, " $7 " GB available"}'
echo "  disk free here: $(df -h . | awk 'NR==2 {print $4}')"
echo
if [ $ok = 1 ]; then echo "All required tools present. Next: make check"; else
cat <<'EOF'
Something is missing. Install hints:
  Ubuntu / Debian / WSL2:  sudo apt update && sudo apt install build-essential openmpi-bin libopenmpi-dev python3-pip
                           pip install numpy pillow matplotlib
  macOS (Homebrew):        brew install gcc open-mpi   (then use: make CC="mpicc" and gcc-14 via OMPI_CC=gcc-14)
  Windows:                 use WSL2 + Ubuntu (recommended)
EOF
exit 1; fi
