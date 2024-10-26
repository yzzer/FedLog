from pydantic import BaseModel


class ServerInfo(BaseModel):
    host: str = ""
    port: int = 0


class FedModel(BaseModel):
    input_model_base64: str = ""
    main_model_base64: str = ""
    output_model_base64: str = ""
    type: str = "fl"