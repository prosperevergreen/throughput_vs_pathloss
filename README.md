# Throughput vs Pathloss

Web app for computing and displaying throughput vs pathloss

## Requirements

To run the application, install [node](https://nodejs.org/en/) version **>=14.0.0** with [npm](https://www.npmjs.com/) version **»–>=8.15.0** and [python](https://www.python.org/) version **>=3.0** must be installed.

### Setup development environment 
To start the development environment via terminal or commandline:
1. To start the client application, run `cd client && npm install && npm start`.
2. In another terminal, start the server with `cd server && pip install -r requirements.txt && python app.py`
3. The app runs on http://localhost:3000.

## Requirements using docker and docker compose

To run the application via docker, [docker engine](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) must be installed. Alternately, [Docker desktop](https://docs.docker.com/compose/install/compose-desktop/) can be simply downloaded which comes with both [Docker](https://docs.docker.com/engine/) and [Docker-compose](https://docs.docker.com/compose/).

### Setup development environment 
To start the development environment:
1. Run `docker-compose up` to start the app in development mode. Docker will need to build the container if it's the first time you run it.
2. The app runs on http://localhost:3000.


### Setup production environment
To start the production environment:
1. Run `docker-compose -f docker-compose.yml up -d` to start the app in production mode. Docker will need to build the container if it's the first time you run it.
2. The app runs on http://localhost:80.
