# Shared factory helpers. Source from other scripts after ROOT is set.
# ROOT must be an absolute path to the repo.

factory_python() {
  local root="${ROOT:?ROOT must be set before factory_python}"
  if [[ -f "$root/.venv/bin/python" ]]; then
    printf '%s\n' "$root/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}
