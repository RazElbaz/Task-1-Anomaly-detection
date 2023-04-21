import ast
import pickletools
import sys
from typing import Optional, TextIO, Tuple
import pickle

import fickling.analysis

if sys.version_info < (3, 9):
    from astunparse import unparse
else:
    from ast import unparse

from fickling.pickle import Interpreter, Pickled

def code(self) -> str:
    """
    Returns the string representation of the code object that was pickled.
    """
    code = self.pickled['code']
    return code.co_code.decode('utf-8')
def check_safety(
    pickled: Pickled,filename, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None
) -> bool:
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr

    properties = pickled.properties
    likely_safe = True
    reported_shortened_code = set()

    def shorten_code(ast_node) -> Tuple[str, bool]:
        code = unparse(ast_node).strip()
        if len(code) > 32:
            cutoff = code.find("(")
            if code[cutoff] == "(":
                shortened_code = f"{code[:code.find('(')].strip()}(...)"
            else:
                shortened_code = code
        else:
            shortened_code = code
        was_already_reported = shortened_code in reported_shortened_code
        reported_shortened_code.add(shortened_code)
        return shortened_code, was_already_reported

    safe_lines = []
    with open(filename, 'rb') as f:
        code=str(f.read().decode('latin1'))
        # print("=====")
        # print(code)
        # print("====")
    for line in code.split('\n'):
        try:
            clean_string = line.replace('\x00', '')
            ast_node = compile(clean_string, '<string>', 'exec', ast.PyCF_ONLY_AST)

        except SyntaxError:
            # If the line doesn't parse, assume it's unsafe and include it in the new file
            # safe_lines.append(line)
            continue
        is_safe = True
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == 'eval'
                ):
                    # eval is always unsafe
                    is_safe = False
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'loads'
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'pickle'
                ):
                    # loading pickles is unsafe, unless it's being done by the check_safety function
                    is_safe = False
            elif isinstance(node, ast.Import):
                # importing modules is unsafe, unless it's a standard library module
                for alias in node.names:
                    if not alias.name.startswith('_') and alias.name not in sys.modules:
                        is_safe = False
            elif isinstance(node, ast.ImportFrom):
                # importing from modules is unsafe, unless it's a standard library module
                if not node.module.startswith('_') and node.module not in sys.modules:
                    is_safe = False
            elif ("eval" in line) or ("exec" in line) or ("compile" in line) or ("open" in line):
                is_safe = False
            elif ("__builtin__"in line) or ("os" in line) or ("subprocess" in line) or ("sys" in line) or ("builtins" in line) or ("socket" in line):
                is_safe = False
        if is_safe:
            safe_lines.append(str(line))
    with open(filename, 'rb') as f:
        data = f.read()

    for op in pickletools.genops(data):
        # print(op)
        if type(op[1]) == str and all(substring not in op[1] for substring in ["eval", "exec", "compile", "open", "__builtin__", "os", "subprocess", "sys", "builtins", "socket"]):
            safe_lines.append(op[1])
            # print("lalallaa"+op[1])

    # write the safe lines to a new pickle file
    with open('safe.pkl', 'wb') as f:
         pickle.dump('\n'.join(safe_lines), f)
    print('lalala')
    print(safe_lines)
    print('lalala')
    if not safe_lines:
        # If there are no safe lines, return False to indicate that the pickle is unsafe
        return False

    if likely_safe:
        stderr.write(
            "Warning: Fickling failed to detect any overtly unsafe code, but the pickle file may "
            "still be unsafe.\n\nDo not unpickle this file if it is from an untrusted source!\n"
        )
        return True
    else:
        return False

filename='unsafe.pkl'
with open(filename, 'rb') as f:
    pickled_data = f.read()
pickled_obj = Pickled.load(pickled_data)
print(fickling.analysis.check_safety(pickled_obj))
check_safety(pickled_obj,filename)
with open('safe.pkl', 'rb') as f:
    pickled_data = f.read()
pickled_obj = Pickled.load(pickled_data)
# print(pickled_data.decode('latin1'))
print(fickling.analysis.check_safety(pickled_obj))

