#!/bin/bash
# set -e
. utils.sh
. setup_options.sh

# Setup log file location:
LOG_FILE="./setup_log.txt"

# Helper functions:
function exit_with_error() {
    echo "Setup finished with errors."
    # echo $1
    # echo $2
    exit 1
}

function exec_and_check() {
    # Run a bash command and check result. Stop script if command fails and $4=1.
    local -r DEFAULT_REQ_SUCESS=0
    local -r REQ_SUCCESS=${4:-${DEFAULT_REQ_SUCESS}}
    if [[ $H4D_HEADLESS_MODE ]]; then
        execute "$1" "$2" "$3"
        exitCode=$?
        # execute() calls print_result() so don't need to do it here
    else
        . "$1"
        exitCode=$?
        print_result "${exitCode}" "$2"
    fi
    if [[ $exitCode -ne 0 && ${REQ_SUCCESS} -eq 1 ]]; then
        exit_with_error "'$2' failed" "${exitCode}"
    fi
}

# Important to deactivate h4d_env before running execute() commands:
# eval "$(conda shell.bash hook)"
# conda deactivate

if [[ "$H4D_HEADLESS_MODE" -eq 1 ]]; then
    rm "${LOG_FILE}"
    # Don't need to delete prev log, the command below creates a new one
    echo "HEADLESS_MODE enabled. To watch logs in realtime open a new terminal and run: "
    echo "    tail -n 15 -F $LOG_FILE"
    echo ""
    echo "===================================================================
Starting Setup
" >"$LOG_FILE"
fi

REQ_SUCCESS=1
DONT_REQ_SUCCESS=0

# We want env and data setup to be successful.
exec_and_check "./setup_env.sh" "Conda Environment Setup" "$LOG_FILE" "$REQ_SUCCESS"
exec_and_check "./setup_datasets.sh" "Datasets Setup" "$LOG_FILE" "$REQ_SUCCESS"
exec_and_check "./setup_coco.sh" "MSCOCO Setup" "$LOG_FILE" "$DONT_REQ_SUCCESS"
exec_and_check "./setup_voc.sh" "PASAL VOC Setup" "$LOG_FILE" "$DONT_REQ_SUCCESS"

# If we get failures on the subsequent steps we can fix the issue and re-run failed ones
# individually/manually:
exec_and_check "./setup_rcnn.sh" "Setup Faster-RCNN" "$LOG_FILE" "$DONT_REQ_SUCCESS"
exec_and_check "./setup_ssd.sh" "Setup SSD" "$LOG_FILE" "$DONT_REQ_SUCCESS"
exec_and_check "./setup_centernet.sh" "Setup CenterNet" "$LOG_FILE" "$DONT_REQ_SUCCESS"
# Old yolo code was giving a lot of problems. We found at least two other potential yolo/pytorch replacements.
# The switch to a new yolo is WIP in the "new_yolo" branch. So disable the old one for now:
#exec_and_check "./setup_yolo.sh" "Setup yolov3" "$LOG_FILE" "$DONT_REQ_SUCCESS"

echo "Setup Complete"
exit 0
