import re
import os
from string import Template

with open("readme_template.md") as readme_file:
    readme = readme_file.read()

codes = re.findall(r'\$code_([^\s]*)', readme)
demos = re.findall(r'\$demo_([^\s]*)', readme)
template_dict = {}

for code_src in codes:
    with open(os.path.join("demo", f"{code_src}.py")) as code_file:
        python_code = code_file.read()
        python_code = python_code.replace('if __name__ == "__main__":\n    iface.launch()', "iface.launch()")
        if python_code.startswith("# Demo"):
            python_code = "\n".join(python_code.split("\n")[2:])
        template_dict[f"code_{code_src}"] = "```python\n" + python_code + "\n```"

for demo_src in demos:
    template_dict[
        f"demo_{demo_src}"
    ] = f"![{demo_src} interface](demo/screenshots/{demo_src}/1.gif)"

readme_template = Template(readme)
output_readme = readme_template.substitute(template_dict)

with open("README.md", "w") as readme_md:
    readme_md.write(output_readme)