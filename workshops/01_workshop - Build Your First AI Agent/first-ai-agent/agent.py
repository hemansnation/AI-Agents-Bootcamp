import os
import google.generativeai as genai
from duckduckgo_search import DDGS

api_key = "Your key"
genai.configure(api_key=api_key)

# model = genai.GenerativeModel('gemini-2.5-flash')
# response = model.generate_content("hi how are you?")
# print(response.text)

# tools - calculator
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# tool - web search

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            summary = "\n".join([f"{r['title']}: {r['body']}" for r in results])
            return summary if summary else "No results found."
    except Exception as e:
        return f"Search Error: {str(e)}"


model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    tools=[calculator, web_search]
)

# agent's conversation loop version 1


chat = model.start_chat(history=[])

# def agent_response(user_input: str) -> str:
#     response = chat.send_message(user_input)

#     if response.candidates[0].content.parts[0].function_call:
#         func_call = response.candidates[0].content.parts[0].function_call
#         if func_call.name == "calculator":
#             args = func_call.args['expression']
#             tool_result = calculator(args)

#             final_response = chat.send_message(
#                 genai.protos.Part(
#                     function_response=genai.protos.FunctionResponse(
#                         name="calculator",
#                         response={'result': tool_result}
#                     )
#                 )
#             )
#             return final_response.text
#     else:
#         return response.text
    
# agent's conversation loop version 2


def agent_response(user_input: str) -> str:
    response = chat.send_message(user_input)
    
    if response.candidates[0].content.parts and response.candidates[0].content.parts[0].function_call:
        func_call = response.candidates[0].content.parts[0].function_call
        func_name = func_call.name
        args = dict(func_call.args)
        tool_result = None

        if func_name == "calculator":
            tool_result = calculator(args['expression'])
        elif func_name == "web_search":
            tool_result = web_search(args['query'])
        
        if tool_result:
            final_response = chat.send_message(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name,
                        response={'result': tool_result}
                    )
                )
            )
            return final_response.text
    else:
        return response.text

# add interactive main loop

if __name__ == "__main__":
    print("Student Helper Agent: Ask me anything! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        reply = agent_response(user_input)
        print(f"Agent: {reply}")

    
