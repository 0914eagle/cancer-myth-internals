# Re-launch the calling script under nohup and return immediately.
#
#   source scripts/lib/detach.sh "<tag>"        # near the top, after DATA_ROOT is known
#
# Anything expected to run longer than half an hour goes through this, so a
# dropped SSH session never kills a run. The environment (DATA_ROOT, CONFIG,
# GPUS, ...) is inherited by the re-launched copy because `VAR=x bash script`
# exports VAR into the child's environment.
#
#   FOREGROUND=1  run in this shell instead (smoke tests, debugging)
#   DETACHED=1    set by this file on the re-launched copy; never set it by hand
#
# Every launch is appended to ${DATA_ROOT}/cancer_myth_internals/logs/jobs.tsv
# (timestamp, pid, tag, log) so `bash scripts/jobs.sh` can list them.

_detach_tag="${1:-job}"
if [[ "${DETACHED:-0}" != "1" && "${FOREGROUND:-0}" != "1" ]]; then
  _detach_root="${DATA_ROOT:-/data1/heejae}/cancer_myth_internals/logs"
  mkdir -p "${_detach_root}"
  _detach_log="${_detach_root}/${_detach_tag}_$(date +%Y%m%d_%H%M%S).log"
  _detach_script="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)/$(basename "${BASH_SOURCE[1]}")"
  DETACHED=1 nohup bash "${_detach_script}" "$@" > "${_detach_log}" 2>&1 &
  _detach_pid=$!
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${_detach_pid}" "${_detach_tag}" "${_detach_log}" >> "${_detach_root}/jobs.tsv"
  echo "[nohup] ${_detach_tag} started, pid ${_detach_pid}"
  echo "[nohup] log: ${_detach_log}"
  echo "[nohup] follow: tail -f ${_detach_log}"
  echo "[nohup] list:   bash scripts/jobs.sh"
  exit 0
fi
unset _detach_tag _detach_root _detach_log _detach_script _detach_pid
