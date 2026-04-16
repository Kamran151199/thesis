#!/bin/bash
# Usage: ./download-transcript.sh <youtube-url> <output-name>
# Example: ./download-transcript.sh "https://www.youtube.com/watch?v=kCc8FmEb1nY" "karpathy-gpt-from-scratch"
#
# This downloads the auto-generated English transcript with timestamps.
# The transcript file lands in transcripts/<output-name>.txt
#
# Install yt-dlp first:  brew install yt-dlp

set -e

URL="$1"
NAME="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/transcripts"

if [ -z "$URL" ] || [ -z "$NAME" ]; then
    echo "Usage: $0 <youtube-url> <output-name>"
    echo "Example: $0 \"https://www.youtube.com/watch?v=kCc8FmEb1nY\" \"karpathy-gpt-from-scratch\""
    exit 1
fi

mkdir -p "$OUT_DIR"

# Download auto-generated subtitles in SRT format, then convert to plain text with timestamps
yt-dlp --write-auto-sub --sub-lang en --skip-download \
    --sub-format srt \
    -o "$OUT_DIR/$NAME" \
    "$URL"

# Convert SRT to readable timestamped text
SRT_FILE="$OUT_DIR/${NAME}.en.srt"
TXT_FILE="$OUT_DIR/${NAME}.txt"

if [ -f "$SRT_FILE" ]; then
    # Parse SRT: extract timestamp + text, skip sequence numbers and blank lines
    awk '
    /^[0-9]+$/ { next }
    /^$/ { next }
    /-->/ {
        split($1, t, "[:,]")
        printf "[%s:%s:%s] ", t[1], t[2], t[3]
        next
    }
    { print }
    ' "$SRT_FILE" > "$TXT_FILE"
    rm "$SRT_FILE"
    echo "Transcript saved to: $TXT_FILE"
else
    # Fallback: check for .vtt
    VTT_FILE="$OUT_DIR/${NAME}.en.vtt"
    if [ -f "$VTT_FILE" ]; then
        # Parse VTT similarly
        awk '
        /^WEBVTT/ { next }
        /^Kind:/ { next }
        /^Language:/ { next }
        /^$/ { next }
        /-->/ {
            split($1, t, "[:,.]")
            printf "[%s:%s:%s] ", t[1], t[2], t[3]
            next
        }
        /^[0-9]+$/ { next }
        { print }
        ' "$VTT_FILE" > "$TXT_FILE"
        rm "$VTT_FILE"
        echo "Transcript saved to: $TXT_FILE"
    else
        echo "No subtitle file found. The video may not have auto-generated captions."
        exit 1
    fi
fi

echo "Done! You can now ask Claude about any timestamp in this video."
