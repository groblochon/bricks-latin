from util.configs import build_generator_function_config
from util.enums import State
from . import euclidean_distance, INPUT_EXAMPLE


def get_config():
    return build_generator_function_config(
        function=euclidean_distance,
        input_example=INPUT_EXAMPLE,
        data_type="text",
        issue_id=230,
        tabler_icon="VectorTriangle",
        min_refinery_version="1.7.0",
        state=State.PUBLIC.value,
        gdpr_compliant="True",
        type="python_function",
        kern_token_proxy_usable="False",
        docker_image="None",
        available_for=["refinery", "common"],
        part_of_group=["distance", "gdpr_compliant"], # first entry should be parent directory
        # bricks integrator information
        integrator_inputs={
            "name": "euclidean_distance",
            "refineryInputType": "text",
            "outputs": ["contains mistakes", "no mistakes"],
            "constants": {
                "inputAttribute": { # previously YOUR_ATTRIBUTE, never optional
                    "selectionType": "string",
                    "defaultValue": "your-text",
                }, 
                "baseSentence": {
                    "selectionType": "string",
                    "defaultValue": "The sun is not made out of pancakes.",
                    "description": "The text you want to compare your records to.",
                    "optional": "False",
                },
            },
        },
    )