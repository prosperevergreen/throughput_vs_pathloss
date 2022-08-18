# Throughput vs Pathloss

Web app for computing and displaying throughput vs pathloss

### Requirements

To run the application, the user needs to have [docker engine](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed. Alternately, [Docker desktop](https://docs.docker.com/compose/install/compose-desktop/) can be simply downloaded which comes with both [Docker](https://docs.docker.com/engine/) and [Docker-compose](https://docs.docker.com/compose/).
### Setup development environment
To start the development environment:
1. Run `docker-compose up` to start the app in development mode. Docker will need to build the container if it's the first time you run it.
2. The app runs on http://localhost:3000.


### Setup production environment
To start the production environment:
1. Run `docker-compose -f docker-compose.yml up -d` to start the app in production mode. Docker will need to build the container if it's the first time you run it.
2. The app runs on http://localhost:80.
