from results import YouTubeCommentsAnalyzer
from extractingcomments import YouTubeCommentsExtractor
from flask import Flask, request, render_template
from onlycomments import OnlyComments
import yt_video

api_key="AIzaSyCcp49l_YL1lisFQqp10ay7xnT1oCSx0qo"

app = Flask(__name__)
analyzer = YouTubeCommentsAnalyzer(api_key)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process_input", methods=["POST"])
def take_input():
    video_id = request.form["query"]
    result = analyzer.get_result(videoID=video_id)
    my_youtube_video = yt_video.YoutubeInput(video_id=video_id)
    
    image_thumbnail_url = my_youtube_video.get_thumbnail()
    video_name = my_youtube_video.yt.title
    positive = result["positive_count"]
    neutral = result["neutral_count"]
    negative = result["negative_count"]
    
    result_list: list = [positive, neutral, negative]
    no_of_comments = sum(result_list)
    p_result_list = [round(count*100/no_of_comments, 2) for count in result_list]

    #summarizer
    summary=analyzer.generate_negative_summary(videoID=video_id)
    return render_template("datavisualization.html",
                           result=p_result_list,
                           result_dict=result,
                           image_thumbnail_url=image_thumbnail_url,
                           video_name=video_name,
                           comments=result_list,
                           negative_comments_summary=summary
                           )

if __name__ == "__main__":
    app.run(debug=True)
