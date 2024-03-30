import pydantic


# класс(-ы), описывающий выход сервиса
class ServiceOutput(pydantic.BaseModel):
    """_summary_

    Args:
        pydantic (_type_): _description_

    Returns:
        _type_: _description_
    """

    width: int = pydantic.Field(default=640)
    height: int = pydantic.Field(default=480)
    channels: int = pydantic.Field(default=3)
