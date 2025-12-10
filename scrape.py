import requests

payload = { 'api_key': 'd61ee9d0fe95444dcbee54f3fb4b3995', 'url': 'https://www.youtube.com/shorts/veZMkE9OaZ8' }
r = requests.get('https://api.scraperapi.com/', params=payload)
print(r.text)
