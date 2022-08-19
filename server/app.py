from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import json

from file_handler import Processor, Plotter

PLOT_IMAGE_NAME = 'plot_image.png'

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return '<h1>This is the throughput-vs-pathloss api server</h1>'


@app.route('/api/throughput-vs-pathloss', methods=['POST'])
def throughput_vs_pathloss():
    # Extract request payload as JSON
    request_data = request.get_json()

    # Get CSVData: [{pathloss: number, pdcp: number },...,]
    oldCSVData = request_data['oldCSVData']
    newCSVData = request_data['newCSVData']

    # Process CSVData
    new_CSV_data_processor = Processor(newCSVData)
    old_CSV_data_processor = Processor(oldCSVData)

    # Extract data in format pdcp: [number, ...,], pathloss: [number, ...,]
    pdcp1, pathloss1 = new_CSV_data_processor.csv_handler()
    pdcp2, pathloss2 = old_CSV_data_processor.csv_handler()

    # Plot graph of throughput vs pathloss
    plot = Plotter(pdcp1, pathloss1, pdcp2, pathloss2)
    print("I ran")
    # Save graph as .png image
    plot.plot_function(PLOT_IMAGE_NAME)

    return send_from_directory('./', PLOT_IMAGE_NAME, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5005)
