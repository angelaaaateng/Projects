from flask import Flask,  redirect, url_for, request
app = Flask(__name__)

@app.route('/')
def hello_pangea():
	return("Welcome Jeremie!")

@app.route('/pangeaapp/<text>')
def hello_world(text):
	return("Hello" +text)

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
	app.debug = True
	app.run()
