from util.configs import build_classifier_function_config
from util.enums import State, RefineryDataType, BricksVariableType, SelectionType
from . import language_detection, INPUT_EXAMPLE


def get_config():
    return build_classifier_function_config(
        function=language_detection,
        input_example=INPUT_EXAMPLE,
        issue_id=1,
        tabler_icon="AlphabetGreek",
        min_refinery_version="1.7.0",
        state=State.PUBLIC.value,
        type="python_function",
        kern_token_proxy_usable="false",
        docker_image="none",
        available_for=["refinery", "common"],
        part_of_group=[
            "text_analysis",
        ],  # first entry should be parent directory
        # bricks integrator information
        integrator_inputs={
            "name": "language_detection",
            "refineryDataType": RefineryDataType.TEXT.value,
            "outputs": [
                "af",
                "ar",
                "bg",
                "bn",
                "ca",
                "cs",
                "cy",
                "da",
                "de",
                "el",
                "en",
                "es",
                "et",
                "fa",
                "fi",
                "fr",
                "gu",
                "he",
                "hi",
                "hr",
                "hu",
                "id",
                "it",
                "ja",
                "kn",
                "ko",
                "lt",
                "lv",
                "mk",
                "ml",
                "mr",
                "ne",
                "nl",
                "no",
                "pa",
                "pl",
                "pt",
                "ro",
                "ru",
                "sk",
                "sl",
                "so",
                "sq",
                "sv",
                "sw",
                "ta",
                "te",
                "th",
                "tl",
                "tr",
                "uk",
                "ur",
                "vi",
                "zh-cn",
                "zh-tw",
            ],
            "variables": {
                "ATTRIBUTE": {
                    "selectionType": SelectionType.CHOICE.value,
                    "addInfo": [
                        BricksVariableType.ATTRIBUTE.value,
                        BricksVariableType.GENERIC_STRING.value,
                    ],
                }
            },
        },
    )
