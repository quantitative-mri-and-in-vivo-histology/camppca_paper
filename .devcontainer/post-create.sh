#!/usr/bin/env bash
# Runs once after the container is created. Must stay idempotent.
set -uo pipefail

# Copy the host's ~/.ssh so permissions can be fixed without touching the host.
if [ -d /home/node/.ssh-host ]; then
  mkdir -p /home/node/.ssh
  cp -r /home/node/.ssh-host/. /home/node/.ssh/ 2>/dev/null || true
  chmod 700 /home/node/.ssh
  chmod 600 /home/node/.ssh/id_* 2>/dev/null || true
  chmod 644 /home/node/.ssh/*.pub 2>/dev/null || true
fi

# camppca is a submodule pinned to one commit. Install it first and editable,
# so this repo's own install finds it already satisfied.
CORE=/workspace/submodules/camppca
if [ -f "$CORE/pyproject.toml" ]; then
  sudo uv pip install --system -e "$CORE" || echo "post-create: WARNING - camppca install failed."
else
  echo "post-create: WARNING - $CORE is empty. Run: git submodule update --init"
fi

if [ -f /workspace/pyproject.toml ]; then
  sudo uv pip install --system -e "/workspace[dev]" || echo "post-create: WARNING - project install failed."
fi

# The sudo installs write root-owned egg-info into the bind mount. Hand it back.
sudo chown -R node:node /workspace/*.egg-info /workspace/src/*.egg-info "$CORE"/*.egg-info "$CORE"/src/*.egg-info /workspace/build 2>/dev/null || true
