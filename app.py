from crypt import methods
from config import app
from flask import jsonify, render_template, request
from inference_code import Datapred
import os


@app.route("/")
def hello():
    return render_template('home.html')


@app.route('/trial')
def test():
    return render_template(('test.html'))


@app.route("/analyze", methods=['POST'])
def analyze():
    if request.method == 'POST':
        sentence = request.form.get('text')
        score = Datapred(sentence)
        print(score)
        return render_template('analyze.html', score=score)
    return "Not built yet"


@app.route('/api', methods=["POST"])
def api():
    if request.method == 'GET':
        return "Unauthorized"
    input_json = request.get_json(force=True)
    sentences = input_json['comments']
    results = []
    for sentence in sentences:
        results.append(Datapred(sentence))
    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
