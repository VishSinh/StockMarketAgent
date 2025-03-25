# Store Market Agent

## How to Use with Docker

1. Create a `.env` file based on `.env.example`:
    ```sh
    cp .env.example .env
    ```

2. Build and run the Docker container using Docker Compose:
    ```sh
    docker-compose up --build
    ```

## APIs

### Get Stock by Ticker
```sh
curl --location 'http://127.0.0.1:5000/stock?ticker=AAPL'
```

### Get Stock by Name
```sh
curl --location 'http://127.0.0.1:5000/stock?stock=Apple'
```
---
### Deployment Test Link
```sh
curl --location 'https://stockmarketagent.onrender.com/stock?stock=Apple'
```
*Note: It may take 50 seconds or more as it takes time to spin up the instance if inactive for long.*
