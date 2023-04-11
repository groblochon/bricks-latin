from util.configs import build_extractor_function_config
from util.enums import State, RefineryDataType, BricksVariableType, SelectionType
from . import price_extraction, INPUT_EXAMPLE


def get_config():
    return build_extractor_function_config(
        function=price_extraction,
        input_example=INPUT_EXAMPLE,
        data_type="text",
        issue_id=37,
        tabler_icon="CurrencyEuro",
        min_refinery_version="1.7.0",
        state=State.PUBLIC.value,
        gdpr_compliant="True",
        type="python_function",
        kern_token_proxy_usable="False",
        docker_image="None",
        available_for=["refinery", "common"],
        part_of_group=["numbers", "gdpr_compliant"], # first entry should be parent directory
        # bricks integrator information
        integrator_inputs={
            "name": "price_extraction",
            "refineryDataType": RefineryDataType.TEXT.value,
            "variables": {
                "ATTRIBUTE": {
                    "selectionType": SelectionType.CHOICE.value,
                    "optional": "false",
                    "addInfo": [
                        BricksVariableType.ATTRIBUTE.value,
                        BricksVariableType.GENERIC_STRING.value
                    ]
                },
                "LABEL": {
                    "selectionType": SelectionType.CHOICE.value,
                    "defaultValue": "price",
                    "optional": "false",
                    "addInfo": [
                        BricksVariableType.GENERIC_STRING.value
                    ]
                }
            }
        }
    )
