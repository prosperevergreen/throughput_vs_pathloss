import flask
import flask_cors
import json

from file_handler import Processor, Plotter

PLOT_IMAGE_NAME = 'plot_image.png'
LINEAR_REGRESSION = 'linear_regression'
POLY_REGRESSION = 'Poly_regression'

app = flask.Flask(__name__)
flask_cors.CORS(app)


@app.route('/')
def home():
    return '<h1>This is the throughput-vs-pathloss api server</h1>'


@app.route('/api/throughput-vs-pathloss', methods=['POST'])
def throughput_vs_pathloss():
    # Extract request payload as JSON
    request_data = flask.request.get_json()
    #print("this is request data:",request_data)
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
    # Save graph as .png image
    plot.plot_function(PLOT_IMAGE_NAME)
    # plot.machine_learning(LINEAR_REGRESSION, POLY_REGRESSION)

    return flask.send_from_directory('./', PLOT_IMAGE_NAME, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5005)
