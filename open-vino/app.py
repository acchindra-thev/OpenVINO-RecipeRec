from flask import Flask, render_template, request
import urllib3
from openVino_predict import run_openvino

app = Flask(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        ingredients = run_openvino()

        recipes = ["Pesto Chicken Bake", "Pasta Pesto with Mozzarella and Tomatoes"]
        images = [
            "https://static.wixstatic.com/media/6db271_e796096026b24636b83f5d861d3fd723~mv2.jpg/v1/crop/x_5,y_0,w_669,h_1020/fill/w_272,h_416,al_c,q_80,usm_0.66_1.00_0.01/Chicken-Pesto-Prep-3_5-3-1.webp",
            "https://static.wixstatic.com/media/6db271_d9a9b3990b474be5b7038b8070e5abbf~mv2.jpg/v1/crop/x_10,y_0,w_1181,h_1800/fill/w_272,h_416,al_c,q_80,usm_0.66_1.00_0.01/pesto-pasta-recipe-5.webp"]
        return render_template('app.html', ingredients=ingredients, recipes=recipes, images=images)

    return render_template('app.html', ingredients=["Upload ingredients to get recommendations!"],
                           recipes=["Upload ingredients to get recommendations!"], images=[])


if __name__ == "__main__":
    app.run(debug=True)
