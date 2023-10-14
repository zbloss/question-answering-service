import requests

query = 'Find me a wine that is both fruity and earthy.'
qa_service_url = 'http://localhost:8000/query-text'

response = requests.post(qa_service_url, params={'payload': query})
print(response.json())