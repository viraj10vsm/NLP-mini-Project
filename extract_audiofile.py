import os
from pytube import YouTube

# YouTube video URL
video_url = 'https://www.youtube.com/watch?v=1vsmaEfbnoE'

# Download the YouTube video
yt = YouTube(video_url)
audio_stream = yt.streams.filter(only_audio=True).first()

# Save the audio stream to a file
audio_file_path = 'C:/Users/viraj/Desktop/yt sentiments sem 6/audio.mp4'  # Name of the downloaded audio file
audio_stream.download(output_path='.', filename='audio.mp4')