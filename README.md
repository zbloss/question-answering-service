# question-answering-service
RAG-based system to do question-answering on a given corpus of text.

## Getting Started

The following tools are required to get up and running with this service:
1. Git
2. Docker
3. docker-compose

To launch the service, execute the following:

1. `git clone https://github.com/zbloss/question-answering-service/tree/main`
2. `docker-compose up`

This launches two services. The first is this launches a Weaviate Vector DB service, and the second is an NLP model service wrapped in a FastAPI deployment.

* The Weaviate Vector DB is an open-source tool that allows for low-latency read and writes of model embeddings (vectors) and associated metadata.
* The NLP Service loads data stored in this git repository, embeds it using the [huggingface/transformers](https://huggingface.co/models) Model of your choice, and batch inserts both the embedded vectors and associated metadata into the Weaviate Vector DB

By default, this project uses a fairly poor-accurate `distilbert-base-uncased` but this can be easily swapped by changing the `MODEL_NAME` environment variable in the `docker-compose.yml` file in the root of this git repository.

* I chose to roll a lower-performing model to ensure maximum compatibility across hardware. For the same reason, you will find code that enables doing this computing on GPU's, but it is not utilized.

## Example API Usage

Also available at `qa-service/api/example_api_call.py`

```python

import requests

query = 'Find me a wine that is both fruity and earthy.'
qa_service_url = 'http://localhost:8000/query-text'

response = requests.post(qa_service_url, params={'payload': query})
print(response.json())

```

## Project Requirements

| Requirement | Complete |
|-----|-----|
| A high-level system diagram showing the different services in your system and how they interact with each other | X |
|Services should be containerized with a docker-compose.yml file for local deployment | X |
| Usage of a vector database for knowledge retrieval | X |
| Usage of either locally deployed (and containerized) models or an API such as OpenAI, Cohere, or Huggingface Hosted Inference API; Usage of an API framework of your choice (Flask, Django, FastAPI, etc.) | X |
| Expose a single endpoint that takes a natural language question and returns a natural language response; | Almost |
| No front-end required, but please provide instructions on how to use your API | X |

Almost all of the requirements are satisfied. The one requirement not completely satisfied requires an interface for both a Natural Language question and a Natural Langauge response. 

This project does allow for Natural Language questions however I was not able to return a Natural Language response in the time I spent working on it. Instead, the user is returned with the Top 3 best matches from the vector database.

Adding a natural language response is something that can easily be implemented in a 2 hours. This would involved standing up a third service in this docker-compose stack. This service would be another FastAPI model service that takes the response from the `qa-service-api` here and provides a human-friendly response back.

## Next Steps

Given another few hours, these are the improvements I would make to this project.

1. Tests. I have a good handful of unit tests for the wrapper code I wrote, but I did not touch the API service yet.
2. An additional Natural Language Inference service that takes the qa-service-api output and makes it more conversational.
3. Kubernetes manifests. This could be rolled into a helm-chart, thereby being made a highly-reusable piece of infrastructure given that the model under the hood is Environment Variable driven.
