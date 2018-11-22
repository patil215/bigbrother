# usage: /bin/bash shrink_video.sh input
ffmpeg -i "${1}" -vf scale="1080:-2" "shrunk.mp4"
