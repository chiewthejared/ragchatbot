from google.ai.generativelanguage_v1beta import ModelServiceClient

client = ModelServiceClient()
for model in client.list_models():
    print(model.name)