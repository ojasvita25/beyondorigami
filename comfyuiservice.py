import websocket 
import uuid
import json
import random
import urllib.request
import urllib.parse

save_image_websocket = 'SaveImageWebsocket'
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def get_prompt_with_workflow(text, image):
    with open("image_image_lcm_api.json", 'r', encoding="utf-8") as f:
        workflow_jsondata = f.read()

    jsonwf = json.loads(workflow_jsondata)

    #set the text prompt for our positive CLIPTextEncode
    jsonwf["6"]["inputs"]["text"] = text

    jsonwf["19"]["inputs"]["image"] = image
    
    seednum = random.randint(0, 2**63 - 1)
    #set the seed for our KSampler node
    jsonwf["3"]["inputs"]["seed"] = seednum
    return jsonwf

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_image = None
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        node_number = data['node']
                        current_node = prompt[node_number]["class_type"]
        else:
            if current_node == save_image_websocket:
                output_image = out[8:]

    return output_image


def fetch_image_from_comfy(prompt, image):
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, get_prompt_with_workflow(prompt, image))
    ws.close()
    return images

