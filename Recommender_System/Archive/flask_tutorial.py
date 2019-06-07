from flask import Flask, render_template
app = Flask(__name__)
# __name__ is just the name of the module
#route decorators - how you get to website pages
#to run this, you can also type "flask run" - environment vars
#https://www.youtube.com/watch?v=MwZwr5Tvyxo&list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH

@app.route("/")
@app.route("/home")
def home():
    return "<h1>Home Page</h1>"

@app.route("/about")
def about():
    return "<h1>About Page</h1>"

#this is just saying we don't need to restart terminal all the time
if __name__ == '__main__':
    app.run(debug=True)
