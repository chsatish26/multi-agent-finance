from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    return {"ok": True, "echo": payload}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
