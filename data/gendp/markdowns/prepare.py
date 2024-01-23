import strip_markdown
import os

#Test sur action_sociale_familles.md
print(os.getcwd()+"/data/gendp/texts")

input_file_path = os.path.join(os.path.dirname(__file__), 'action_sociale_familles.md')

str = strip_markdown.strip_markdown_file(input_file_path, text_fn= os.getcwd()+"/data/gendp/texts")