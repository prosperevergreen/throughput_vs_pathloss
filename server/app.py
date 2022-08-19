from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import json

from file_handler import Processor


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '<h1>This is the throughput-vs-pathloss api server</h1>'


@app.route('/api/throughput-vs-pathloss', methods=['POST', 'OPTION'])
def throughput_vs_pathloss():
    request_data = request.get_json()
    oldCSVData = request_data['oldCSVData']
    newCSVData = request_data['newCSVData']
    new_CSV_data_processor = Processor(newCSVData)
    old_CSV_data_processor = Processor(oldCSVData)
    new_CSV_data_processor.csv_handler()
    old_CSV_data_processor.csv_handler()
    print(request.method)

    return jsonify({"oldCSVData": oldCSVData, "newCSVData": newCSVData})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
