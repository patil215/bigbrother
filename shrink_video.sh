# usage: /bin/bash shrink_video.sh input
ffmpeg -i "${1}" -vf scale="720:-1" "small_${1}"
