# usage: /bin/bash shrink_video.sh input
ffmpeg -i "${1}" -vf scale="720:-2" "shrunk.mp4"
