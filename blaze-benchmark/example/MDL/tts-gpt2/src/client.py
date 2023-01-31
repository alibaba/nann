import requests
import time

entity_ids = 592772768984
params={
    "entity_id":entity_ids,
    "title":'dress'
}
cnt = 5
start = time.time()
for i in range(cnt):
    t = requests.get("http://127.0.0.1/rttm/dp", timeout=600, params=params)
    print(t.text)
end = time.time()
print('avg time:', (end-start)/cnt, 's')
