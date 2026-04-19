'''
README:

这个class是模仿VLM-AD这个论文的free/structured prompt设计的，主要是为了让LLM更好地理解输入的内容和任务要求。
这个prompt包含了system message和user message两部分

'''
'''
Prompt Engineering in a nutshell:

System message是给LLM的角色设定和背景信息，告诉它应该以什么样的身份来回答问题。
比如在这个例子中，
User message是给LLM的具体任务描述和输入内容，告诉它需要完成什么样的任务。
比如在这个例子中，

你可以把System message看成是LLM的全局“角色设定”，告诉它应该以什么样的身份来回答问题。
你可以把User message看成是LLM的本地“任务描述”，告诉它这次推理需要完成什么样的任务。

优先级自然是System message > User message，因为System message是全局的角色设定，会影响LLM的整体行为和回答风格，而User message只是针对当前任务的具体描述。

一些常见的trick包括在System message中加入关于输出格式的要求，比如要求LLM输出JSON格式，或者要求LLM输出特定的字段，这样可以让LLM更好地理解和遵循输出要求。
同时也方便做后续下游任务的处理，比如解析LLM的输出，或者把LLM的输出作为其他模型的输入。
'''

FREEDOM_SYS_LEGACY = (
    "You are an expert in autonomous driving. "
    "This is the front-view image of the ego vehicle. "
    "When explaining the reasoning, please focus on the camera image and the surrounding environment"
)

FREEDOM_USER_LEGACY = (
    "1. Please describe the ego vehicle’s current actions.\n"
    "2. Please predict the ego vehicle’s future actions.\n"
    "3. Please explain the reasoning behind both the current and future actions."
)


STRUCTURED_SYS_LEGACY = (
    "You are an expert in autonomous driving. "
    "This is the front-view image of the ego vehicle. "
)

STRUCTURED_USER_LEGACY = (
    "Please describe the ego vehicle’s action from the control action list: {go straight, move slowly, stop, reverse}.\n"
    "Please describe the ego vehicle’s action from the turn action list: {turn left, turn right, turn around, none}.\n"
    "Please describe the ego vehicle’s action from the lane action list: {change lane to the left, change lane to the right, merge into the left lane, merge into the right lane, none}"
)


FREEDOM_SYS = (
    "You are the ego vehicle's cautious driver-assistant, describing what the car should do from a real driving perspective. "
    "You must ground every claim in visible evidence from the front-view image. "
    "Avoid generic safety slogans and avoid repeating the same statement."
)

FREEDOM_USER = (
    "Task:\n"
    "1. Describe the ego vehicle's most likely current action.\n"
    "2. Predict the ego vehicle's immediate next action (next 2-5 seconds).\n"
    "3. Provide concise reasoning based on visible cues (lane geometry, lead vehicles, pedestrians, traffic lights/signs, obstacles, road edge).\n\n"
    "Output rules:\n"
    "- Keep the answer concise and information-dense.\n"
    "- No filler text, no disclaimers, no self-reference.\n"
    "- Use exactly this format:\n"
    "Current action: <one sentence>\n"
    "Next action: <one sentence>\n"
    "Reasoning: <2-4 short bullet points with concrete visual evidence>"
)


STRUCTURED_SYS = (
    "You are the ego vehicle's control module. "
    "Your job is to output only machine-readable action flags from the front-view image."
)

STRUCTURED_USER = (
    "Choose exactly one flag from each list below.\n"
    "control_flag: {go straight, move slowly, stop, reverse}\n"
    "turn_flag: {turn left, turn right, turn around, none}\n"
    "lane_flag: {change lane to the left, change lane to the right, merge into the left lane, merge into the right lane, none}\n\n"
    "Return ONLY valid JSON in one line, with no extra text and no markdown:\n"
    '{"control_flag":"<one option>","turn_flag":"<one option>","lane_flag":"<one option>"}'
)

class Prompt:
    def __init__(self, system_message:str = None, user_message:str = None, seed:str = ""):
        seed_upper = seed.upper() if seed else ""
        if seed_upper == "FREEDOM":
            system_message = FREEDOM_SYS
            user_message = FREEDOM_USER
        elif seed_upper == "STRUCTURED":
            system_message = STRUCTURED_SYS
            user_message = STRUCTURED_USER

        assert system_message is not None and user_message is not None, (
            "Either seed must be provided or both system_message and user_message must be provided."
        )
        self.system_message = system_message
        self.user_message = user_message

    def __str__(self):
        return f"System Message: {self.system_message}\nUser Message: {self.user_message}"
