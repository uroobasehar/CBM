# Source with an absolute packed prefix as the first argument.
# conda-pack's upstream activation script expects optional CONDA_PREFIX/PS1
# variables and probes for `deactivate`; preserve strict caller options around it.
cbm_activate_packed_prefix() {
    local cbm_prefix="$1"
    local cbm_options="$-"
    local cbm_status
    set +u
    set +e
    source "$cbm_prefix/bin/activate"
    cbm_status=$?
    case "$cbm_options" in *u*) set -u ;; esac
    case "$cbm_options" in *e*) set -e ;; esac
    return "$cbm_status"
}
cbm_activate_packed_prefix "$1"
unset -f cbm_activate_packed_prefix
