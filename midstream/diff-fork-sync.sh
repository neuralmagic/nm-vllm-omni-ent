#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_URL="https://github.com/vllm-project/vllm-omni"
FORK_URL="https://github.com/neuralmagic/nm-vllm-omni-ent"

usage() {
    cat <<EOF
Usage: diff-fork-sync.sh [OPTIONS]

Compare fork-specific patches between two upstream releases to identify
which files actually need review in a sync PR.

  Upstream: ${UPSTREAM_URL}
  Fork:     ${FORK_URL}

Required:
  --new-upstream TAG   New upstream release tag (e.g. v0.26.0)
  --old-fork BRANCH    Fork branch based on old upstream (e.g. main)
  --new-fork BRANCH    Fork branch for the sync PR (e.g. sync-v0.26.0)

Optional:
  --old-upstream TAG   Previous upstream base. If omitted, auto-detected
                       via merge-base between old-fork and upstream/main.
  --output-dir DIR     Output directory for summary and patches (default: sync-review/)
  --verbose            List dropped and identical files (default: counts only)
  --force              Overwrite output directory if it already exists
  --help               Show this help

Example (reviewing the v0.24.0 → v0.26.0 sync PR):
./diff-fork-sync.sh \
    --new-upstream v0.26.0 \
    --old-fork main \
    --new-fork sync-v0.26.0 \
    --output-dir /tmp/sync-review
EOF
    exit "${1:-0}"
}

parse_args() {
    OLD_UPSTREAM=""
    NEW_UPSTREAM=""
    OLD_FORK=""
    NEW_FORK=""
    OUTPUT_DIR="sync-review"
    VERBOSE=false
    FORCE=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --old-upstream) OLD_UPSTREAM="$2"; shift 2 ;;
            --new-upstream) NEW_UPSTREAM="$2"; shift 2 ;;
            --old-fork)     OLD_FORK="$2"; shift 2 ;;
            --new-fork)     NEW_FORK="$2"; shift 2 ;;
            --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
            --verbose|-v)   VERBOSE=true; shift ;;
            --force|-f)     FORCE=true; shift ;;
            --help|-h)      usage 0 ;;
            *)              echo "Unknown option: $1" >&2; usage 1 ;;
        esac
    done

    if [[ -z "$NEW_UPSTREAM" || -z "$OLD_FORK" || -z "$NEW_FORK" ]]; then
        echo "Error: --new-upstream, --old-fork, and --new-fork are required." >&2
        usage 1
    fi
}

find_remote_for_url() {
    local target="$1" name url
    while IFS=$'\t' read -r name url; do
        url="${url% (fetch)}"
        url="${url%.git}"
        if [[ "$url" == "$target" ]]; then
            echo "$name"
            return
        fi
    done < <(git remote -v | grep '(fetch)')
    return 1
}

ensure_remotes() {
    UPSTREAM_REMOTE=$(find_remote_for_url "$UPSTREAM_URL") || {
        echo "No remote found for ${UPSTREAM_URL}"
        echo "Adding remote 'upstream-omni'..."
        git remote add upstream-omni "${UPSTREAM_URL}.git"
        UPSTREAM_REMOTE="upstream-omni"
    }
    FORK_REMOTE=$(find_remote_for_url "$FORK_URL") || {
        echo "No remote found for ${FORK_URL}"
        echo "Adding remote 'nm-vllm-omni-ent'..."
        git remote add nm-vllm-omni-ent "${FORK_URL}.git"
        FORK_REMOTE="nm-vllm-omni-ent"
    }
    echo "Using remotes: upstream=${UPSTREAM_REMOTE}, fork=${FORK_REMOTE}"
}

resolve_refs() {
    NEW_UPSTREAM_REF="$NEW_UPSTREAM"
    OLD_FORK_REF="${FORK_REMOTE}/${OLD_FORK}"
    NEW_FORK_REF="${FORK_REMOTE}/${NEW_FORK}"

    if [[ -n "$OLD_UPSTREAM" ]]; then
        OLD_UPSTREAM_REF="$OLD_UPSTREAM"
    else
        OLD_UPSTREAM_REF=$(git merge-base "$OLD_FORK_REF" "${UPSTREAM_REMOTE}/main" 2>/dev/null) || {
            echo "Error: could not auto-detect old upstream base." >&2
            echo "  Specify --old-upstream explicitly." >&2
            exit 1
        }
        local short
        short=$(git log --oneline -1 "$OLD_UPSTREAM_REF")
        echo "Auto-detected old upstream base: ${short}"
    fi

    for ref in "$OLD_UPSTREAM_REF" "$NEW_UPSTREAM_REF" "$OLD_FORK_REF" "$NEW_FORK_REF"; do
        if ! git rev-parse --verify "$ref" &>/dev/null; then
            echo "Error: ref '$ref' not found. Run 'git fetch' or check your remote/tag names." >&2
            exit 1
        fi
    done
}

fork_patch() {
    git diff "$1...$2" -- "$3" 2>/dev/null
}

normalize_patch() {
    grep -v -E '^(index [0-9a-f]|@@ |diff --git |--- |(\+\+\+) )' | cut -c2- || true
}

classify_files() {
    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf '$tmpdir'" EXIT

    git diff --name-only "${OLD_UPSTREAM_REF}...${OLD_FORK_REF}" | sort > "$tmpdir/old.txt"
    git diff --name-only "${NEW_UPSTREAM_REF}...${NEW_FORK_REF}" | sort > "$tmpdir/new.txt"

    NEW_FILES=$(comm -13 "$tmpdir/old.txt" "$tmpdir/new.txt")
    DROPPED_FILES=$(comm -23 "$tmpdir/old.txt" "$tmpdir/new.txt")
    local common
    common=$(comm -12 "$tmpdir/old.txt" "$tmpdir/new.txt")

    CHANGED_FILES=""
    UNCHANGED_FILES=""
    while IFS= read -r f || [[ -n "$f" ]]; do
        [[ -z "$f" ]] && continue
        if diff \
            <(fork_patch "$OLD_UPSTREAM_REF" "$OLD_FORK_REF" "$f" | normalize_patch) \
            <(fork_patch "$NEW_UPSTREAM_REF" "$NEW_FORK_REF" "$f" | normalize_patch) \
            &>/dev/null; then
            UNCHANGED_FILES+="$f"$'\n'
        else
            CHANGED_FILES+="$f"$'\n'
        fi
    done <<< "$common"
}

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'

count_lines() {
    [[ -z "$1" ]] && echo 0 && return
    echo -n "$1" | grep -c . || echo 0
}

print_section() {
    local color="$1" label="$2" files="$3"
    [[ -z "$files" ]] && return 0
    local n
    n=$(count_lines "$files")
    echo "${color}${BOLD}${label} (${n}):${RESET}"
    while IFS= read -r f || [[ -n "$f" ]]; do
        [[ -n "$f" ]] && echo "  ${DIM}-${RESET} $f"
    done <<< "$files"
    echo ""
}


write_file_list() {
    local files="$1"
    [[ -z "$files" ]] && return 0
    while IFS= read -r f || [[ -n "$f" ]]; do
        [[ -n "$f" ]] && echo "- \`$f\`"
    done <<< "$files"
    return 0
}

write_summary() {
    local summary="${OUTPUT_DIR}/summary.md"
    local review_count=$(( $(count_lines "$CHANGED_FILES") + $(count_lines "$NEW_FILES") ))

    cat > "$summary" <<HEADER
# Fork Sync Review: ${OLD_UPSTREAM_REF} → ${NEW_UPSTREAM}

**${review_count} files need review**

## Patch files

- **all.patch** — The fork's current patch against \`${NEW_UPSTREAM}\`. Shows what the fork adds on top of the new upstream for every CHANGED and NEW file. Use this to understand the full fork delta.
- **delta.patch** — Patch-of-patches for CHANGED files only. Shows how the fork's delta *shifted* between releases. Use this to spot conflict resolution errors — lines that appear here are what actually changed during the sync.

HEADER

    if [[ -n "$CHANGED_FILES" ]]; then
        echo "## Patch changed ($(count_lines "$CHANGED_FILES"))" >> "$summary"
        echo "" >> "$summary"
        echo "Fork-specific patch differs between releases. Review \`delta.patch\` for what changed, \`all.patch\` for the full current state." >> "$summary"
        echo "" >> "$summary"
        write_file_list "$CHANGED_FILES" >> "$summary"
        echo "" >> "$summary"
    fi

    if [[ -n "$NEW_FILES" ]]; then
        echo "## New fork files ($(count_lines "$NEW_FILES"))" >> "$summary"
        echo "" >> "$summary"
        echo "New fork-specific changes not present in the previous release. See \`all.patch\` for content." >> "$summary"
        echo "" >> "$summary"
        write_file_list "$NEW_FILES" >> "$summary"
        echo "" >> "$summary"
    fi

    echo "## Dropped from fork ($(count_lines "$DROPPED_FILES"))" >> "$summary"
    echo "" >> "$summary"
    echo "Were in the old fork delta but not in the new one. Likely upstreamed." >> "$summary"
    echo "" >> "$summary"
    if [[ -n "$DROPPED_FILES" ]]; then
        write_file_list "$DROPPED_FILES" >> "$summary"
        echo "" >> "$summary"
    fi

    echo "## Patch identical ($(count_lines "$UNCHANGED_FILES"))" >> "$summary"
    echo "" >> "$summary"
    echo "Fork patch carried over unchanged. No review needed." >> "$summary"
    echo "" >> "$summary"
    if [[ -n "$UNCHANGED_FILES" ]]; then
        write_file_list "$UNCHANGED_FILES" >> "$summary"
        echo "" >> "$summary"
    fi
}

write_combined_patch() {
    local combined="${OUTPUT_DIR}/all.patch"
    local separator="================================================================"

    write_patch_section() {
        local label="$1" files="$2" upstream_ref="$3" fork_ref="$4"
        [[ -z "$files" ]] && return 0
        while IFS= read -r f || [[ -n "$f" ]]; do
            [[ -z "$f" ]] && continue
            echo "$separator" >> "$combined"
            echo "# $label: $f" >> "$combined"
            echo "$separator" >> "$combined"
            fork_patch "$upstream_ref" "$fork_ref" "$f" >> "$combined"
            echo "" >> "$combined"
        done <<< "$files"
        return 0
    }

    > "$combined"
    write_patch_section "CHANGED" "$CHANGED_FILES" "$NEW_UPSTREAM_REF" "$NEW_FORK_REF"
    write_patch_section "NEW"     "$NEW_FILES"     "$NEW_UPSTREAM_REF" "$NEW_FORK_REF"
}

write_delta_patch() {
    local delta="${OUTPUT_DIR}/delta.patch"
    local separator="================================================================"

    cat > "$delta" <<'HEADER'
# Delta Patch
#
# What changed in the fork's patch between releases.
# Standard unified diff: - lines were in the old fork patch,
# + lines are in the new fork patch.

HEADER
    [[ -z "$CHANGED_FILES" ]] && return 0
    while IFS= read -r f || [[ -n "$f" ]]; do
        [[ -z "$f" ]] && continue
        echo "$separator" >> "$delta"
        echo "# CHANGED: $f" >> "$delta"
        echo "$separator" >> "$delta"
        diff -u \
            --label "a/$f (old fork patch)" \
            --label "b/$f (new fork patch)" \
            <(fork_patch "$OLD_UPSTREAM_REF" "$OLD_FORK_REF" "$f" | normalize_patch) \
            <(fork_patch "$NEW_UPSTREAM_REF" "$NEW_FORK_REF" "$f" | normalize_patch) \
            >> "$delta" || true
        echo "" >> "$delta"
    done <<< "$CHANGED_FILES"
    return 0
}

write_output_dir() {
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    write_summary
    write_combined_patch
    write_delta_patch
    echo "Wrote output to ${OUTPUT_DIR}/"
    echo ""
}

print_report() {
    local review_count=0

    echo "${BOLD}${CYAN}========================================================================${RESET}"
    echo "${BOLD}  Fork Sync Review: ${OLD_UPSTREAM_REF} → ${NEW_UPSTREAM}${RESET}"
    echo "${BOLD}${CYAN}========================================================================${RESET}"
    echo ""

    print_section "$RED"    "REVIEW  — patch changed"      "$CHANGED_FILES"
    print_section "$RED"    "REVIEW  — new fork files"      "$NEW_FILES"
    if $VERBOSE; then
        print_section "$YELLOW" "VERIFY  — dropped from fork"   "$DROPPED_FILES"
        print_section "$GREEN"  "SKIP    — patch identical"      "$UNCHANGED_FILES"
    else
        echo "${YELLOW}${BOLD}VERIFY${RESET}  — dropped from fork ($(count_lines "$DROPPED_FILES"))"
        echo "${GREEN}${BOLD}SKIP${RESET}    — patch identical ($(count_lines "$UNCHANGED_FILES"))"
    fi
    echo ""

    review_count=$(( $(count_lines "$CHANGED_FILES") + $(count_lines "$NEW_FILES") ))

    echo "${BOLD}${CYAN}========================================================================${RESET}"
    echo "${BOLD}  ${review_count} files need review${RESET}"
    echo "${DIM}  ${OUTPUT_DIR}/all.patch   — full fork patches (CHANGED + NEW)${RESET}"
    echo "${DIM}  ${OUTPUT_DIR}/delta.patch — patch-of-patches (CHANGED only)${RESET}"
    echo "${DIM}  ${OUTPUT_DIR}/summary.md  — file classification + descriptions${RESET}"
    echo "${BOLD}${CYAN}========================================================================${RESET}"
}

main() {
    local orig_dir script_dir repo_root
    orig_dir="$(pwd)"
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)" || {
        echo "Error: could not find git repo containing this script." >&2
        exit 1
    }
    cd "$repo_root"

    parse_args "$@"

    # Resolve output dir relative to where the script was invoked
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="${orig_dir}/${OUTPUT_DIR}"
    fi

    if [[ -e "$OUTPUT_DIR" ]] && ! $FORCE; then
        echo "Error: '${OUTPUT_DIR}' already exists. Use --force to overwrite." >&2
        exit 1
    fi

    ensure_remotes
    resolve_refs

    classify_files
    write_output_dir
    print_report
}

main "$@"
