import os
from yt_dlp import YoutubeDL


url = "https://youtube.com/shorts/vSX3IRxGnNY"
output_file = "input.mp4"

ydl_opts = {
    'outtmpl': output_file,
    'format': 'bestvideo+bestaudio/best',
    'merge_output_format': 'mp4'
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])