import subprocess
import time
from openai import OpenAI, APIConnectionError

def start_server(model_path, port):
    # Start vLLM server as a subprocess
    server_process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/".join(model_path.split("/")[1:]),
        "--port", str(port)
    ])
    while True:
        retries = 20
        try:
            # Create OpenAI-compatible client
            client = OpenAI(
                base_url=f"http://localhost:{port}/v1",
                api_key="not-needed"
            )
            # Make a request
            completion = client.chat.completions.create(
                model=model_path,
                messages=[
                    {"role": "user", "content": "Hello!"}
                ]
            )
            print("Response:", completion.choices[0].message.content)
            break
        except APIConnectionError as e:
            if retries == 0:
                raise(e)
            print("Sleeping for 10 seconds...")
            time.sleep(10)
            retries -= 1
            print(f"Retries left: {retries}")
        except Exception as e:
            close_server()
    return server_process

def close_server(server_process):
    # Terminate a running vLLM server
    server_process.terminate()
    return server_process.wait()