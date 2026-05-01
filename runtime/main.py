
import sys
import os

# Add the app dir to sys.path for deps
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    return {"ok": True, "payload": payload}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
