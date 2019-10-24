
#!/bin/bash
function abs_path {
	cd "$1"
	loc="$(pwd)"
	echo "$loc"
}

answer_is_yes() {
    [[ "$REPLY" =~ ^[Yy]$ ]] \
        && return 0 \
        || return 1
}

ask() {
    print_question "$1"
    read -r
}

ask_for_confirmation() {
    print_question "$1 (y/n) "
    read -r -n 1
    printf "\n"
}

print_success() {
    printf " [✔] %s\n" "$1"
}

print_warning() {
    print_in_yellow "   [!] $1\n"
}

print_error() {
    printf " [✖] %s\n" "$1"
}

print_error_stream() {
    while read -r line; do
        print_error "↳ ERROR: $line"
    done
}

print_result() {
    # params: Call execute() and its exitCode to print_result(), along with a descriptor for what command was attempted.
    #   $1 (exitCode): exit code of command that was run
    #   $2 (msg): A short (<1 line, or <40-60 chars) description of the command.
    # if exit code is
    if [ "$1" -eq 0 ]; then
        print_success "$2"
    else
        print_error "$2"
    fi

    return "$1"
}

set_trap() {
    trap -p "$1" | grep "$2" &> /dev/null \
        || trap '$2' "$1"
}

print_in_color() {
    printf "%b" \
        "$(tput setaf "$2" 2> /dev/null)" \
        "$1" \
        "$(tput sgr0 2> /dev/null)"
}

print_in_green() {
    print_in_color "$1" 2
}

print_in_purple() {
    print_in_color "$1" 5
}

print_in_red() {
    print_in_color "$1" 1
}

print_in_yellow() {
    print_in_color "$1" 3
}

print_question() {
    print_in_yellow "   [?] $1"
}


execute() {
    # params:
    # $1: command to run
    # $2: message to display (short, one line description of command)
    # $3: path to log file to use (if not specified, it goes to a /tmp/XXXX file)

    local -r CUSTOM_LOG=$3
    local exitCode=0
    local cmdsPID=""
    local -r CMDS="$1"
    local -r MSG="${2:-${1}}"
    local LOGFILE="/tmp/XXXXX" # original
    LOGFILE="${3:-${LOGFILE}}" # try to write to local log

    if [[ $3 ]]; then
        # echo "custom logfile specified"
        local -r TMP_FILE=$LOGFILE
    else
        # echo "custom logfile not specified"
        local -r TMP_FILE="$(mktemp $LOGFILE)"
    fi

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # If the current process is ended,
    # also end all its subprocesses.

    set_trap "EXIT" "kill_all_subprocesses"

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Execute commands in background

    if [[ $CUSTOM_LOG ]]; then
        # new:
        eval "$CMDS" 1>> $TMP_FILE 2>&1 &
        cmdsPID=$!
    else
        # original:
        eval "$CMDS" &> /dev/null 2> $TMP_FILE &
        cmdsPID=$!
    fi

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Show a spinner if the commands
    # require more time to complete.

    show_spinner "$cmdsPID" "$CMDS" "$MSG"

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Wait for the commands to no longer be executing
    # in the background, and then get their exit code.

    # original:
    # wait "$cmdsPID" &> /dev/null
    # new:
    if [[ $CUSTOM_LOG ]]; then
        wait "$cmdsPID" 1>> $TMP_FILE 2>&1
        # wait "$cmdsPID" &> $TMP_FILE
    else
        wait "$cmdsPID" &> $TMP_FILE
    fi
    exitCode=$?

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Print output based on what happened.

    print_result $exitCode "$MSG"

    if [ $exitCode -ne 0 ]; then
        if [[ $CUSTOM_LOG ]]; then
            print_error_stream < "$TMP_FILE"
        else
            print_error_stream < "$TMP_FILE"
        fi
    fi

    # rm -rf "$TMP_FILE"

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return $exitCode

}


show_spinner() {

    local -r FRAMES='/-\|'

    # shellcheck disable=SC2034
    local -r NUMBER_OR_FRAMES=${#FRAMES}

    local -r CMDS="$2"
    local -r MSG="$3"
    local -r PID="$1"

    local i=0
    local frameText=""

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Note: In order for the Travis CI site to display
    # things correctly, it needs special treatment, hence,
    # the "is Travis CI?" checks.

    if [ "$TRAVIS" != "true" ]; then

        # Provide more space so that the text hopefully
        # doesn't reach the bottom line of the terminal window.
        #
        # This is a workaround for escape sequences not tracking
        # the buffer position (accounting for scrolling).
        #
        # See also: https://unix.stackexchange.com/a/278888

        printf "\n\n\n"
        tput cuu 3

        tput sc

    fi

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Display spinner while the commands are being executed.

    while kill -0 "$PID" &>/dev/null; do

        frameText="   [${FRAMES:i++%NUMBER_OR_FRAMES:1}] $MSG"

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Print frame text.

        if [ "$TRAVIS" != "true" ]; then
            printf "%s\n" "$frameText"
        else
            printf "%s" "$frameText"
        fi

        sleep 0.2

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Clear frame text.

        if [ "$TRAVIS" != "true" ]; then
            tput rc
        else
            printf "\r"
        fi

    done

}
OS="$(uname)"
function dlcmd() {
    if [[ "${OS}" == "Linux" ]]; then
        wget -c "$1" -O "$3"
    else
        curl -o "$3" "$1"
    fi  
}

