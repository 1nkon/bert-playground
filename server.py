from flask import Flask
from flask import request
from flask import abort
from flask import Response
import bertft

app = Flask(__name__)

pipeline = bertft.Pipeline()


@app.route('/', methods=['POST'])
def hello_world():
    if not request.is_json:
        abort(Response("Json expected"))

    sentence = request.json['sentence']
    data = pipeline.do_find(sentence)
    response = app.response_class(
        response=data.to_json(orient='table', index=False),
        status=200,
        mimetype='application/json'
    )
    return response
