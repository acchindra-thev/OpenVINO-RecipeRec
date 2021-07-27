from flask import Flask, render_template, request
import urllib3
import requests
import json
from openVino_predict import run_openvino

app = Flask(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@app.route('/', methods=['POST', 'GET'])
def main():

    # recipes[i] corresponds to images[i] corresponds to missed_ing_nums_[i] ... etc.
    recipes = []
    images = []
    missed_ingredient_numbers = []

    if request.method == 'POST':
        num_recipes_to_show = 5
        ignore_pantry = True
        sorting_priority = 1
        ingredients = run_openvino()

        recipe_json = CallAPI(ingredients, num_recipes_to_show, ignore_pantry, sorting_priority)

        for recipe in recipe_json:
            recipes.append(recipe['title'])
            images.append(recipe['image'])
            missed_ingredient_numbers.append(recipe['missedIngredientCount'])

        # images = [
        #     "https://static.wixstatic.com/media/6db271_e796096026b24636b83f5d861d3fd723~mv2.jpg/v1/crop/x_5,y_0,w_669,h_1020/fill/w_272,h_416,al_c,q_80,usm_0.66_1.00_0.01/Chicken-Pesto-Prep-3_5-3-1.webp",
        #     "https://static.wixstatic.com/media/6db271_d9a9b3990b474be5b7038b8070e5abbf~mv2.jpg/v1/crop/x_10,y_0,w_1181,h_1800/fill/w_272,h_416,al_c,q_80,usm_0.66_1.00_0.01/pesto-pasta-recipe-5.webp"
        # ]
        return render_template('app.html', ingredients=ingredients, recipes=recipes, images=images, missed_ingredient_numbers = missed_ingredient_numbers)

    return render_template('app.html', ingredients=["Upload ingredients to get recommendations!"],
                           recipes=["Upload ingredients to get recommendations!"], images=images, missed_ingredient_numbers = missed_ingredient_numbers)


"""
Send a GET request to Spoonacular API, and return recipes that use the specified ingredients
@:param ingredients, a list of ingredients outputted by the image classification model
@:param num_recipes_to_show, user-specified number of recipes to return
@:param ignore_pantry, bool value whether to ignore pantry ingredients (salt, water, etc) or not
@:param sorting_priority, whether to maximize used ingredients (1) or minimize missing ingredients (2) first
@:return recipe_json, a json formatted list of recipes and associated metadata
"""
def CallAPI(ingredients, num_recipes_to_show, ignore_pantry, sorting_priority):
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/findByIngredients"

    querystring = {"ingredients": ingredients,
                   "number": num_recipes_to_show,
                   "ignorePantry": ignore_pantry,
                   "ranking": sorting_priority
                   }

    headers = {
        'x-rapidapi-key': "82ef15bdc3msh893d3386c0b40d6p1939bajsn7d7255d714f2",
        'x-rapidapi-host': "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    #print(response.text)
    recipe_json = json.loads(response.text)
    return recipe_json

if __name__ == "__main__":
    app.run(debug=True)
