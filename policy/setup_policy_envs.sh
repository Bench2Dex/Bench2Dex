#!/usr/bin/env bash
set -u

# Create one conda environment per policy directory, excluding ACT.
# Environment names intentionally match policy directory names.
#
# By default this creates lightweight base environments with Python + pip.
# Set INSTALL_DEPS=1 to also try policy-specific dependency installs where a
# requirements.txt or pyproject.toml is present. Large VLA stacks can take a
# long time and may need manual fixes, so dependency installation is opt-in.

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_RC="${HOME}/.condarc"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "[mirror] Configuring conda to prefer Tsinghua mirrors..."
if [ -f "${CONDA_RC}" ] && [ ! -f "${CONDA_RC}.bak-before-policy-envs" ]; then
    cp "${CONDA_RC}" "${CONDA_RC}.bak-before-policy-envs"
    echo "[mirror] Backed up existing .condarc to ${CONDA_RC}.bak-before-policy-envs"
fi

cat > "${CONDA_RC}" <<'EOF'
channels:
  - defaults
show_channel_urls: true
channel_priority: flexible
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

echo "[mirror] Configuring pip to use ${PIP_INDEX_URL}..."
python -m pip config set global.index-url "${PIP_INDEX_URL}" >/dev/null || true
python -m pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn" >/dev/null || true

env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

create_env() {
    local env_name="$1"
    local py_version="$2"

    if env_exists "${env_name}"; then
        echo "[skip] ${env_name} already exists"
        return 0
    fi

    echo "[create] ${env_name} with python=${py_version}"
    conda create -y -n "${env_name}" "python=${py_version}" pip
}

install_optional_deps() {
    local env_name="$1"
    local policy_dir="${POLICY_ROOT}/${env_name}"

    if [ "${INSTALL_DEPS:-0}" != "1" ]; then
        return 0
    fi

    echo "[deps] Installing optional dependencies for ${env_name}"
    case "${env_name}" in
        GO1|RDT)
            if [ -f "${policy_dir}/requirements.txt" ]; then
                conda run -n "${env_name}" python -m pip install -r "${policy_dir}/requirements.txt"
            fi
            ;;
        GR00T_n15)
            if [ -f "${policy_dir}/pyproject.toml" ]; then
                conda run -n "${env_name}" python -m pip install -e "${policy_dir}[base]"
            fi
            ;;
        DP|openvla-oft|pi0|pi05)
            if [ -f "${policy_dir}/pyproject.toml" ]; then
                conda run -n "${env_name}" python -m pip install -e "${policy_dir}"
            fi
            ;;
        DP3)
            if [ -f "${policy_dir}/3D-Diffusion-Policy/setup.py" ]; then
                conda run -n "${env_name}" python -m pip install -e "${policy_dir}/3D-Diffusion-Policy"
            fi
            ;;
        DexVLA|TinyVLA)
            if [ -f "${policy_dir}/requirements.txt" ]; then
                conda run -n "${env_name}" python -m pip install -r "${policy_dir}/requirements.txt"
            fi
            if [ -f "${policy_dir}/policy_heads/setup.py" ]; then
                conda run -n "${env_name}" python -m pip install -e "${policy_dir}/policy_heads"
            fi
            ;;
        *)
            echo "[deps] No dependency recipe for ${env_name}"
            ;;
    esac
}

# Keep names identical to policy directories. ACT is intentionally excluded.
create_env "DP" "3.9" && install_optional_deps "DP"
create_env "DP3" "3.9" && install_optional_deps "DP3"
create_env "DexVLA" "3.10" && install_optional_deps "DexVLA"
create_env "GR00T_n15" "3.10" && install_optional_deps "GR00T_n15"
create_env "GO1" "3.10" && install_optional_deps "GO1"
create_env "LLaVA-VLA" "3.10" && install_optional_deps "LLaVA-VLA"
create_env "RDT" "3.10" && install_optional_deps "RDT"
create_env "TinyVLA" "3.10" && install_optional_deps "TinyVLA"
create_env "Your_Policy" "3.10" && install_optional_deps "Your_Policy"
create_env "openvla-oft" "3.10" && install_optional_deps "openvla-oft"
create_env "pi0" "3.11" && install_optional_deps "pi0"
create_env "pi05" "3.11" && install_optional_deps "pi05"

echo
echo "[done] Current policy environments:"
conda env list | grep -E '^(DP|DP3|DexVLA|GR00T_n15|GO1|LLaVA-VLA|RDT|TinyVLA|Your_Policy|openvla-oft|pi0|pi05)[[:space:]]'
