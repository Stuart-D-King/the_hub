from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('pylda.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, threaded=True)
