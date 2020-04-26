from flask import Flask
from flask import request
from flask import abort
from flask import Response
from bertft import Pipeline

app = Flask(__name__)

pipeline = Pipeline()


@app.route('/', methods=['POST'])
def hello_world():
    if not request.is_json:
        abort(Response("Json expected"))

    json = request.json
    sentence = json['sentence']
    data = pipeline.do_find(sentence)
    if 'simple' not in json or not json['simple']:
        json_data = data.to_json(orient='table', index=False)
    else:
        json_data = data['word'].to_json(orient='values')
    return app.response_class(response=json_data, status=200, mimetype='application/json')
