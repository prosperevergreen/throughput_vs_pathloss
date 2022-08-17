from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '<h1>This is the throughput-vs-pathloss api server</h1>'


@app.route('/api/throughput-vs-pathloss', methods=['POST'])
def throughput_vs_pathloss():
    request_data = request.get_json()
    oldCSVData = request_data['oldCSVData']
    newCSVData = request_data['newCSVData']
    SSPowerBlock = request_data['SSPowerBlock']
    return jsonify({"oldCSVData": oldCSVData, "newCSVData": newCSVData, "SSPowerBlock": SSPowerBlock})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
