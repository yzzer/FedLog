from pydantic import BaseModel


class ServerInfo(BaseModel):
    host: str = ""
    port: int = 0


class FedModel(BaseModel):
    input_model_base64: str = None
    main_model_base64: str = None
    output_model_base64: str = None
    type: str = "fl"