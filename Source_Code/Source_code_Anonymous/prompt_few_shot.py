from langchain_core.prompts import PromptTemplate

expertise = "usability testing"
cat_users = "Doctor specialized in rhinocytology"
system_name = "anonymous"
system_desc = "It is a web application for rhinocytogram analysis that uses a deep neural network to automatically identify and classify cells from a nasal cytological preparation, based on a digital image taken directly from a microscope."

REQUEST = (
       f"""You are an expert in system {expertise}. Your task is to generate tasks for a user test performed by participants belonging to the category specified by the label {cat_users} in the system description message. Each task must meet the following characteristics:
    - Provide clear instructions on what participants should do
    - Avoid including words that appear in the product interface
    - Do not specify detailed steps to complete the task (e.g., avoid naming specific links or referencing link text)
    - Be atomic, meaning each task must have a single, clear goal

    To generate the tasks, follow these steps:
    1. Analyze the system description to be tested as indicated by the {system_desc} label in the system description message
    2. Identify the system's core functionalities, excluding typical features for its category (e.g., login or registration for web apps)
    3. Generate tasks that test the main functionalities of the system
    4. Verify that the generated tasks meet the specified requirements.
    
    You must explore the screeshots of the user interfaces, analyzing all of their elements before generating the tasks. 

    Examples of tasks that can be employed in this usability study are:
    1. Check the classification of class ‘‘muciparous’’ cells and correct errors, if any.
    2. Identify cells misclassified in class ‘‘ciliated’’ and correct misclassifications and explanations, if any.
"""
)

  
CONTEXT = (f"""The system that is being tested is {system_name}. {system_desc}. The system's main functions include:
    - Automatic cell identification and classification: The system detects various types of cells such as artifacts, bacteria, red blood cells, eosinophils, epithelial cells, ciliated cells, lymphocytes, mast cells, metaplastic cells, mucus-producing cells, and neutrophils.
    - Display of test results: It provides a table showing the type of cell, the number of detected cells, the reference range, and a classification of their quantity.
    - Access to classified cell images: It allows visualization of the classified cell images, categorized by confidence level (low, medium, high).
    - Validation and correction of classifications: It allows the doctor to mark the classification of cells as “Correct” or “Incorrect” and, if incorrect, to correct it by selecting a new class and providing reasons why the original classification was wrong. These features support the doctor in reviewing and potentially correcting the automatic classifications provided by the system. 
           
    The objective of the usability study is to determine whether the rhinocytologist is able to validate and eventually correct explanations in order to reconfigure the model.
"""
)

STRUCTURE = ("""
    The output must be divided in two sections: the first must be labeled as "Reasoning", which must contain the reasoning process that took you to generate the tasks, the second must be labeled "Tasks" and must contain the tasks in the following form:
             - <task1>
             - <task2>
             - ... 
""")


TEMPLATE = "\n\n".join([
    "[Request]\n" + REQUEST + "\n",
    "[Context]\n" + CONTEXT + "\n",
    "[Screenshots]\n{image}",
    "[Structure]\n" + STRUCTURE + "\n",
])

prompt_template = PromptTemplate.from_template(TEMPLATE)

prompt_text = "\n\n".join([
    "[Request]\n" + REQUEST + "\n",
    "[Context]\n" + CONTEXT + "\n",
    "[Structure]\n" + STRUCTURE + "\n"])